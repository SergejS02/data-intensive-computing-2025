#!/usr/bin/env bash
# build & deploy + start S3 watcher so pytest-integration just works
set -euo pipefail
export AWS_PAGER=""

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Lambda names, roles, etc.
PY="python3.11"
ROLE="arn:aws:iam::000000000000:role/lambda-role"
FUNCS=(preprocess profanity sentiment)
L_DIR="src/lambdas"
TMP="/tmp/lambda_build.$$"

BUCKET="reviews-input"
REV_T="Reviews"
USR_T="Users"

# 1) Ensure bucket
awslocal s3api head-bucket --bucket "$BUCKET" 2>/dev/null \
  || awslocal s3 mb "s3://$BUCKET" >/dev/null
# purge
awslocal s3 rm "s3://$BUCKET" --recursive >/dev/null 2>&1 || true

# 2) Tables
for T in "$REV_T" "$USR_T"; do
  KEY="review_id"
  [[ $T == "$USR_T" ]] && KEY="userId"
  awslocal dynamodb describe-table --table-name "$T" >/dev/null 2>&1 || \
    awslocal dynamodb create-table \
      --table-name "$T" \
      --attribute-definitions AttributeName=$KEY,AttributeType=S \
      --key-schema AttributeName=$KEY,KeyType=HASH \
      --billing-mode PAY_PER_REQUEST >/dev/null
  # purge items
  for K in $(awslocal dynamodb scan --table-name "$T" \
             --projection-expression $KEY --output json \
           | jq -r ".Items[].$KEY.S"); do
    awslocal dynamodb delete-item --table-name "$T" \
      --key "{\"$KEY\":{\"S\":\"$K\"}}" >/dev/null
  done
done

# 3) enable stream on Reviews
awslocal dynamodb update-table \
  --table-name "$REV_T" \
  --stream-specification StreamEnabled=true,StreamViewType=NEW_IMAGE \
  >/dev/null 2>&1 || true

STREAM_ARN=$(awslocal dynamodb describe-table --table-name "$REV_T" \
              --query 'Table.LatestStreamArn' --output text)

# 4) SSM params
awslocal ssm put-parameter --name "/dic2025/bucket/reviews" \
  --type String --overwrite --value "$BUCKET"
awslocal ssm put-parameter --name "/dic2025/tables/reviews" \
  --type String --overwrite --value "$REV_T"
awslocal ssm put-parameter --name "/dic2025/tables/users" \
  --type String --overwrite --value "$USR_T"

# 5) Build & deploy Lambdas
rm -rf "$TMP" && mkdir -p "$TMP"
for FN in "${FUNCS[@]}"; do
  SRC="$L_DIR/$FN"
  BLD="$TMP/$FN"
  rm -rf "$BLD" && mkdir -p "$BLD"
  cp "$SRC"/handler.py "$BLD"/

  case $FN in
    preprocess)  pip install -q nltk regex       -t "$BLD" ;;
    profanity)   pip install -q profanityfilter  -t "$BLD" ;;
    sentiment)   pip install -q nltk             -t "$BLD" ;;
  esac

  # bundle corpora
  BUILD_DIR="$BLD" $PY - <<PY >/dev/null 2>&1
import nltk, pathlib, os
d = pathlib.Path(os.environ["BUILD_DIR"]) / "nltk_data"
d.mkdir(exist_ok=True, parents=True)
if "$FN"=="preprocess":
    for c in ("stopwords","wordnet"): nltk.download(c,quiet=True,download_dir=str(d))
else:
    nltk.download("vader_lexicon",quiet=True,download_dir=str(d))
PY

  # absolute zip path
  ZIP="$ROOT/$SRC/lambda.zip"
  mkdir -p "$(dirname "$ZIP")"
  ( cd "$BLD" && zip -qr "$ZIP" . )

  awslocal lambda get-function --function-name "$FN" >/dev/null 2>&1 \
    && awslocal lambda update-function-code \
         --function-name "$FN" --zip-file "fileb://$ZIP" >/dev/null \
    || awslocal lambda create-function \
         --function-name "$FN" \
         --runtime "$PY" \
         --handler handler.handler \
         --zip-file "fileb://$ZIP" \
         --role "$ROLE" \
         --timeout 30 >/dev/null

  awslocal lambda update-function-configuration \
    --function-name "$FN" \
    --environment "Variables={REVIEWS_TABLE=$REV_T,USERS_TABLE=$USR_T}" \
    >/dev/null
done
#ENDPOINT=http://localhost:4566,

# 6) Add S3 → Lambda trigger for "preprocess"
awslocal lambda add-permission \
  --function-name preprocess \
  --statement-id s3invoke \
  --action "lambda:InvokeFunction" \
  --principal s3.amazonaws.com \
  --source-arn arn:aws:s3:::$BUCKET \
  >/dev/null 2>&1 || true

# Ensure S3 triggers Lambda on new object upload
awslocal s3api put-bucket-notification-configuration \
  --bucket $BUCKET \
  --notification-configuration "{
    \"LambdaFunctionConfigurations\": [
      {
        \"LambdaFunctionArn\": \"arn:aws:lambda:us-east-1:000000000000:function:preprocess\",
        \"Events\": [\"s3:ObjectCreated:*\"]
      }
    ]
  }"

# Create DynamoDB → profanity-check stream mapping
awslocal lambda add-permission \
  --function-name profanity \
  --principal dynamodb.amazonaws.com \
  --statement-id dynamodb-trigger-profanity \
  --action lambda:InvokeFunction || true

# 7) Stream → downstream (batch=1)
for FN in profanity sentiment; do
  UUIDS=$(awslocal lambda list-event-source-mappings --function-name "$FN" \
    --query 'EventSourceMappings[].UUID' --output text)

  if [[ -n "$UUIDS" && "$UUIDS" != "None" ]]; then
    for uuid in $UUIDS; do
      awslocal lambda delete-event-source-mapping --uuid "$uuid" 

      for i in {1..5}; do
        sleep 1
        EXISTS=$(awslocal lambda list-event-source-mappings --function-name "$FN" \
          --query "EventSourceMappings[?UUID=='$uuid']" --output json | jq length)
        if [[ "$EXISTS" -eq 0 ]]; then
          break
        fi
      done
    done
  fi
  awslocal lambda create-event-source-mapping \
    --function-name "$FN" \
    --event-source-arn "$STREAM_ARN" \
    --starting-position LATEST \
    --batch-size 1 >/dev/null
done
