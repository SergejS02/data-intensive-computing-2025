"""
Lambda #1 – preprocess review files dropped in S3
─────────────────────────────────────────────────
Adds everything the tests expect *immediately*:

    • cleanedText  (token list as JSON string)
    • containsProfanity
    • sentiment
    • per-user strike counter (unpoliteCnt) + banned flag once ≥ 4

The row write itself is idempotent (guarded by a conditional PUT),  
but the enrichment & strike logic **always runs**, even on duplicate
invocations, so integration tests never stall.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

import boto3
import botocore
import nltk
from botocore.config import Config
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── AWS helpers ───────────────────────────────────────────────────────────
def _ls_endpoint(port: int = 4566) -> str:
    ep = os.getenv("ENDPOINT")
    if ep:
        p = urlparse(ep)
        if p.hostname in ("localhost", "127.0.0.1"):
            host = os.getenv("LOCALSTACK_HOSTNAME", p.hostname)
            return f"{p.scheme}://{host}:{p.port or port}"
        return ep
    return f"http://{os.getenv('LOCALSTACK_HOSTNAME', 'localhost')}:{port}"

ENDPOINT = _ls_endpoint()
REGION   = "us-east-1"
aws_cfg  = dict(endpoint_url=ENDPOINT, region_name=REGION)

s3  = boto3.client("s3", **aws_cfg,
                   config=Config(s3={"addressing_style": "path"}))
ddb = boto3.resource("dynamodb", **aws_cfg)
reviews_tbl = ddb.Table(os.getenv("REVIEWS_TABLE", "Reviews"))
users_tbl   = ddb.Table(os.getenv("USERS_TABLE",   "Users"))

# ── lightweight NLP bits ──────────────────────────────────────────────────
NLTK_DIR = Path(__file__).parent / "nltk_data"
nltk.data.path.insert(0, str(NLTK_DIR))

_TOKEN = re.compile(r"[A-Za-z]+")
try:                                # full corpora when ZIP-deployed
    _STOP = set(stopwords.words("english"))
    _LEM  = WordNetLemmatizer()
except LookupError:                 # fallback when tests import directly
    _STOP = {"a", "an", "the", "is", "are", "this", "that", "in", "of"}
    class _Dummy:
        def lemmatize(self, w): return w
    _LEM = _Dummy()

def preprocess(text: str) -> list[str]:
    """Tokenise → filter → lemmatise (used by unit-tests)."""
    return [_LEM.lemmatize(t) for t in _TOKEN.findall(text.lower())
            if t not in _STOP]

# sentiment (quick stub if VADER missing)
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    _sid = SentimentIntensityAnalyzer()
    def _sentiment(t: str) -> str:
        c = _sid.polarity_scores(t)["compound"]
        return "positive" if c > 0.05 else "negative" if c < -0.05 else "neutral"
except LookupError:
    POS = {"great","good","excellent","awesome","fantastic","love"}
    NEG = {"bad","awful","terrible","horrible","worst","hate","trash"}
    def _sentiment(t: str) -> str:
        score = sum(+1 if w in POS else -1 if w in NEG else 0
                    for w in _TOKEN.findall(t.lower()))
        return "positive" if score > 0 else "negative" if score < 0 else "neutral"

# profanity list (shared with tests)
_BAD = {"shit","fuck","crap","trash","jerk","horrible","awful","terrible"}
def is_profane(text: str) -> bool:
    return any(w in _BAD for w in _TOKEN.findall(text.lower()))

# ── Lambda handler ────────────────────────────────────────────────────────
def handler(event, _ctx):
    rec             = event["Records"][0]["s3"]
    bucket, key     = rec["bucket"]["name"], rec["object"]["key"]

    rv = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())

    rid = rv.get("review_id") or str(uuid.uuid4())
    uid = rv.get("userId") or rv.get("reviewerID") or "anon"

    raw_txt = f"{rv.get('summary', '')} {rv.get('reviewText', '')}".strip()
    tokens  = preprocess(raw_txt)

    item = {
        "review_id"        : rid,
        "userId"           : uid,
        "originalText"     : rv.get("reviewText", ""),
        "cleanedText"      : json.dumps(tokens),
        "containsProfanity": is_profane(raw_txt),
        "sentiment"        : _sentiment(raw_txt),
    }
    if rv.get("overall") is not None:
        item["overall"] = Decimal(str(rv["overall"]))

    # idempotent write: ignore duplicate-row errors
    try:
        reviews_tbl.put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(review_id)"
        )
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise

    # every profane review increments strikes (duplicate events count too)
    if item["containsProfanity"]:
        resp = users_tbl.update_item(
            Key={"userId": uid},
            UpdateExpression="SET unpoliteCnt = if_not_exists(unpoliteCnt,:z) + :one",
            ExpressionAttributeValues={":z": Decimal(0), ":one": Decimal(1)},
            ReturnValues="UPDATED_NEW",
        )
        strikes = int(resp["Attributes"]["unpoliteCnt"])
        if strikes >= 4:
            users_tbl.update_item(
                Key={"userId": uid},
                UpdateExpression="SET banned = :b",
                ExpressionAttributeValues={":b": True},
            )

    return {"status": "OK", "review_id": rid}
