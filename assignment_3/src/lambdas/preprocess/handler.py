"""
Lambda #1 – preprocess a single review JSON uploaded to S3.

• Trigger : s3:ObjectCreated on bucket “reviews-input”
• Action  : write ONE well-formed item into DynamoDB table *Reviews*
            (no NULLs, numbers as Decimal, cleanedText as STRING)
"""

from __future__ import annotations

import json
import os
import re
import uuid
from decimal import Decimal
from pathlib import Path

import boto3
import nltk
from botocore.config import Config
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ────────────────────────────── helpers ───────────────────────────────────
from urllib.parse import urlparse

def _ls_endpoint(port: int = 4566) -> str:
    """
    Resolve the correct LocalStack endpoint inside Lambda.

    • If ENDPOINT is set and NOT localhost → trust it (unit-tests).
    • If ENDPOINT is localhost → swap host for LOCALSTACK_HOSTNAME.
    • Otherwise derive from LOCALSTACK_HOSTNAME or default to localhost.
    """
    if (ep := os.getenv("ENDPOINT")):
        p = urlparse(ep)
        if p.hostname in ("localhost", "127.0.0.1"):
            host = os.getenv("LOCALSTACK_HOSTNAME", p.hostname)
            return f"{p.scheme}://{host}:{p.port or port}"
        return ep

    host = os.getenv("LOCALSTACK_HOSTNAME", "localhost")
    return f"http://{host}:{port}"



# ─────────────────────────── AWS clients ──────────────────────────────────
ENDPOINT = _ls_endpoint()
REGION   = "us-east-1"                            # fine for LocalStack

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    region_name=REGION,
    config=Config(s3={"addressing_style": "path"}),  # required by LocalStack
)
ddb = boto3.resource("dynamodb", endpoint_url=ENDPOINT, region_name=REGION)
tbl = ddb.Table(os.getenv("REVIEWS_TABLE", "Reviews"))

# ───────────────────────────── NLP utils ──────────────────────────────────
NLTK_DIR = Path(__file__).parent / "nltk_data"
nltk.data.path.insert(0, str(NLTK_DIR))

_TOKEN_RE = re.compile(r"\b[A-Za-z]+\b")
_STOP     = set(stopwords.words("english"))
_LEM      = WordNetLemmatizer()


def preprocess(text: str) -> list[str]:
    """Lower-case → tokenize → stop-word filter → lemmatise."""
    return [
        _LEM.lemmatize(tok)
        for tok in _TOKEN_RE.findall(text.lower())
        if tok not in _STOP
    ]


# ───────────────────────────── handler ────────────────────────────────────
def handler(event, _ctx):
    # 1.  S3 notification → bucket & key
    rec     = event["Records"][0]["s3"]
    bucket  = rec["bucket"]["name"]
    key     = rec["object"]["key"]

    # 2.  download review JSON
    raw     = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    review  = json.loads(raw.decode())

    # 3.  derive / normalise fields
    review_id = review.get("review_id") or str(uuid.uuid4())
    user_id   = review.get("userId") or review.get("reviewerID") or "anon"

    text   = f'{review.get("summary", "")} {review.get("reviewText", "")}'.strip()
    tokens = preprocess(text)

    # 4.  assemble DynamoDB item (NO nulls!)
    item: dict[str, object] = {
        "review_id"   : review_id,
        "userId"      : user_id,
        "originalText": review.get("reviewText", ""),
        "cleanedText" : json.dumps(tokens),           # stored as STRING
    }
    if review.get("overall") is not None:
        item["overall"] = Decimal(str(review["overall"]))

    # 5.  put
    tbl.put_item(Item=item)
    return {"status": "OK", "review_id": review_id}
