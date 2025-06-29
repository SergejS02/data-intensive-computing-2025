#!/usr/bin/env bash
# Build & (re)deploy the review-pipeline Lambdas to LocalStack
# and wire all S3 / DynamoDB events.
set -euo pipefail
export AWS_PAGER=""

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── configuration ──────────────────────────────────────────────────────────
PY_RUNTIME="python3.11"
ROLE_ARN="arn:aws:iam::000000000000:role/lambda-role"
FUNCTIONS=(preprocess profanity sentiment)
LAMBDA_DIR="src/lambdas"
TMP_BUILD="/tmp/lambda_build.$$"

REVIEWS_BUCKET="reviews-input"
REVIEWS_TABLE="Reviews"
USERS_TABLE="Users"
ENDPOINT_URL="http://localhost:4566"

# ── prerequisites ──────────────────────────────────────────────────────────
command -v awslocal >/dev/null 2>&1 || { echo "❌  install awscli-local"; exit 1; }
command -v jq      >/dev/null 2>&1 || { echo "❌  install jq";           exit 1; }

echo "🔍  Checking LocalStack …"
awslocal --version >/dev/null
echo "✅  LocalStack is up."

# ── S3 bucket (idempotent) ────────────────────────────────────────────────
if ! awslocal s3api head-bucket --bucket "$REVIEWS_BUCKET" >/dev/null 2>&1; then
  awslocal s3 mb "s3://$REVIEWS_BUCKET" >/dev/null
fi
echo "🧹  Emptying $REVIEWS_BUCKET …"
awslocal s3 rm "s3://$REVIEWS_BUCKET" --recursive >/dev/null 2>&1 || true
awslocal s3api put-bucket-notification-configuration \
  --bucket "$REVIEWS_BUCKET" \
  --notification-configuration '{}' >/dev/null      # clear old rules

# ── DynamoDB tables ───────────────────────────────────────────────────────
for TBL in "$REVIEWS_TABLE" "$USERS_TABLE"; do
  KEY_ATTR=review_id
  [[ $TBL == "$USERS_TABLE" ]] && KEY_ATTR=userId

  awslocal dynamodb describe-table --table-name "$TBL" >/dev/null 2>&1 || \
    awslocal dynamodb create-table \
      --table-name "$TBL" \
      --attribute-definitions AttributeName=$KEY_ATTR,AttributeType=S \
      --key-schema           AttributeName=$KEY_ATTR,KeyType=HASH \
      --billing-mode PAY_PER_REQUEST >/dev/null

  # purge rows so tests start with a clean table
  # … earlier code unchanged …

  # purge rows so tests start with a clean table
  mapfile -t ROWS < <(
    awslocal dynamodb scan --table-name "$TBL" \
      --projection-expression "$KEY_ATTR" \
      --query "Items[].${KEY_ATTR}.S" --output text
  )
  for id in "${ROWS[@]}"; do
    id=$(echo "$id" | xargs)            # ← trim whitespace
    [[ -z "$id" ]] && continue          # ← skip empties
    awslocal dynamodb delete-item \
      --table-name "$TBL" \
      --key "{\"$KEY_ATTR\":{\"S\":\"$id\"}}" >/dev/null || true
  done

done

# stream NEW_IMAGE on Reviews
awslocal dynamodb update-table \
  --table-name "$REVIEWS_TABLE" \
  --stream-specification StreamEnabled=true,StreamViewType=NEW_IMAGE >/dev/null 2>&1 || true
STREAM_ARN=$(awslocal dynamodb describe-table \
  --table-name "$REVIEWS_TABLE" \
  --query "Table.LatestStreamArn" --output text)

# ── minimal SSM parameters (tests need them) ──────────────────────────────
awslocal ssm put-parameter --name "/dic2025/bucket/reviews" --type String --overwrite \
  --value "$REVIEWS_BUCKET" >/dev/null
awslocal ssm put-parameter --name "/dic2025/tables/reviews" --type String --overwrite \
  --value "$REVIEWS_TABLE" >/dev/null
awslocal ssm put-parameter --name "/dic2025/tables/users"   --type String --overwrite \
  --value "$USERS_TABLE" >/dev/null

# ── build & (re)deploy every Lambda ───────────────────────────────────────
rm -rf "$TMP_BUILD" && mkdir -p "$TMP_BUILD"

for FN in "${FUNCTIONS[@]}"; do
  echo "🔧  Building $FN …"
  SRC="$LAMBDA_DIR/$FN"
  BLD="$TMP_BUILD/$FN"
  rm -rf "$BLD" && mkdir -p "$BLD"
  cp "$SRC/handler.py" "$BLD/"

  case "$FN" in
    preprocess) pip install -q nltk regex       -t "$BLD" ;;
    profanity)  pip install -q profanityfilter  -t "$BLD" ;;
    sentiment)  pip install -q nltk             -t "$BLD" ;;
  esac

  # bundle the minimal NLTK bits we need
  BUILD_DIR="$BLD" python3 - <<'PY' >/dev/null 2>&1
import nltk, pathlib, os, sys, json
d = pathlib.Path(os.environ["BUILD_DIR"]) / "nltk_data"
d.mkdir(parents=True, exist_ok=True)
fn = pathlib.Path(os.environ["BUILD_DIR"]).parent.name
if fn == "preprocess":
    for c in ("stopwords", "wordnet"): nltk.download(c, quiet=True, download_dir=str(d))
else:
    nltk.download("vader_lexicon", quiet=True, download_dir=str(d))
PY

  ZIP="$ROOT/$SRC/lambda.zip"
  mkdir -p "$(dirname "$ZIP")"
  ( cd "$BLD" && zip -qr "$ZIP" . )

  if awslocal lambda get-function --function-name "$FN" >/dev/null 2>&1; then
    awslocal lambda update-function-code \
      --function-name "$FN" \
      --zip-file "fileb://$ZIP" >/dev/null
  else
    awslocal lambda create-function \
      --function-name "$FN" \
      --runtime "$PY_RUNTIME" \
      --handler handler.handler \
      --zip-file "fileb://$ZIP" \
      --role "$ROLE_ARN" \
      --timeout 30 >/dev/null
  fi

  # inject runtime env-vars (single line – *no* None placeholders!)
  awslocal lambda update-function-configuration \
    --function-name "$FN" \
    --environment "Variables={ENDPOINT=$ENDPOINT_URL,REVIEWS_TABLE=$REVIEWS_TABLE,USERS_TABLE=$USERS_TABLE}" \
    >/dev/null
done

# ── wire DynamoDB stream → profanity & sentiment ──────────────────────────
for FN in profanity sentiment; do
  # delete old mappings
  awslocal lambda list-event-source-mappings --function-name "$FN" \
    --query "EventSourceMappings[].UUID" --output text |
  xargs -r -n1 awslocal lambda delete-event-source-mapping --uuid >/dev/null

  awslocal lambda create-event-source-mapping \
    --function-name "$FN" \
    --event-source-arn "$STREAM_ARN" \
    --starting-position LATEST \
    --batch-size 1 >/dev/null
done

# ── wire S3 → preprocess ──────────────────────────────────────────────────
PRE_ARN=$(awslocal lambda get-function \
  --function-name preprocess \
  --query "Configuration.FunctionArn" --output text)

awslocal s3api put-bucket-notification-configuration \
  --bucket "$REVIEWS_BUCKET" \
  --notification-configuration \
  "{\"LambdaFunctionConfigurations\":[{\"LambdaFunctionArn\":\"$PRE_ARN\",\"Events\":[\"s3:ObjectCreated:*\"]}]}" \
  >/dev/null

awslocal lambda remove-permission --function-name preprocess --statement-id S3Invoke \
  >/dev/null 2>&1 || true
awslocal lambda add-permission \
  --function-name preprocess \
  --statement-id S3Invoke \
  --action lambda:InvokeFunction \
  --principal s3.amazonaws.com \
  --source-arn "arn:aws:s3:::${REVIEWS_BUCKET}" >/dev/null

# ── tiny watchdog: synthesise S3 events if LocalStack ever drops them ─────
cat > /tmp/watch_s3.sh <<'EOSH'
#!/usr/bin/env bash
set -euo pipefail
BUCKET="reviews-input"; FUNC="preprocess"
SEEN="/tmp/seen.$$"; touch "$SEEN"
while true; do
  mapfile -t KEYS < <(
    awslocal s3api list-objects-v2 --bucket "$BUCKET" \
      --query 'Contents[].Key' --output json | jq -r '.[]')
  for k in "${KEYS[@]}"; do
    grep -Fxq "$k" "$SEEN" && continue
    PAYLOAD=$(cat <<JSON
{ "Records": [ { "s3": { "bucket": { "name": "$BUCKET" },
                          "object": { "key": "$k" } } } ] }
JSON
)
    awslocal lambda invoke --function-name "$FUNC" --payload "$PAYLOAD" /dev/stdout >/dev/null
    echo "$k" >> "$SEEN"
  done
  sleep 0.5
done
EOSH

chmod +x /tmp/watch_s3.sh
nohup /tmp/watch_s3.sh >/dev/null 2>&1 &

echo -e "\n🏁  Deployment complete – run:  pytest tests/\n"
