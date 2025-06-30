from __future__ import annotations
import json
import os
import re
import uuid
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse
import boto3
import botocore.exceptions
import nltk
from botocore.config import Config
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#aws helpers for the connection to localstack
def _ls_endpoint(port: int = 4566) -> str:
    ep = os.getenv("ENDPOINT")
    if ep:
        p = urlparse(ep)
        if p.hostname in ("localhost", "127.0.0.1"):
            host = os.getenv("LOCALSTACK_HOSTNAME", p.hostname)
            return f"{p.scheme}://{host}:{p.port or port}"
        return ep
    return f"http://{os.getenv('LOCALSTACK_HOSTNAME', 'localhost')}:{port}"


aws_cfg = dict(endpoint_url=_ls_endpoint(), region_name="us-east-1")
s3 = boto3.client("s3", **aws_cfg, config=Config(s3={"addressing_style": "path"}))
ddb = boto3.resource("dynamodb", **aws_cfg)
reviews_tbl = ddb.Table(os.getenv("REVIEWS_TABLE", "Reviews"))

#preflagging
NLTK_DIR = Path(__file__).parent / "nltk_data"
nltk.data.path.insert(0, str(NLTK_DIR))

_TOKEN = re.compile(r"[A-Za-z]+")
try:
    _STOP = set(stopwords.words("english"))
    _LEM = WordNetLemmatizer()
except LookupError:                     
    _STOP = {"a", "an", "the", "is", "are", "this", "that", "in", "of"}

    class _Dummy:
        def lemmatize(self, w):  
            return w
    _LEM = _Dummy()


def preprocess(text: str) -> list[str]:
    """Tokenise, stop-word filter, lemmatise."""
    return [
        _LEM.lemmatize(tok)
        for tok in _TOKEN.findall(text.lower())
        if tok not in _STOP
    ]


try:
    from nltk.sentiment import SentimentIntensityAnalyzer

    _sid = SentimentIntensityAnalyzer()

    def _sentiment(t: str) -> str:
        c = _sid.polarity_scores(t)["compound"]
        return "positive" if c > 0.05 else "negative" if c < -0.05 else "neutral"

except LookupError:  #preflagging
    _POS = {"great", "good", "excellent", "awesome", "fantastic", "love"}
    _NEG = {"bad", "awful", "terrible", "horrible", "worst", "hate", "trash"}

    def _sentiment(t: str) -> str:
        score = sum(
            +1 if w in _POS else -1 if w in _NEG else 0
            for w in _TOKEN.findall(t.lower())
        )
        return "positive" if score > 0 else "negative" if score < 0 else "neutral"


_BAD = {"shit", "fuck", "crap", "trash", "jerk", "horrible", "awful", "terrible"}


def is_profane(text: str) -> bool:
    return any(w in _BAD for w in _TOKEN.findall(text.lower()))


#handler logic 
def handler(event, _ctx):
    # exactly one S3 record per invocation
    s3rec = event["Records"][0]["s3"]
    bucket, key = s3rec["bucket"]["name"], s3rec["object"]["key"]

    rv = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    rid = str(uuid.uuid4())
    uid = rv.get("reviewerID") or "anon"

    raw_txt = f"{rv.get('summary', '')} {rv.get('reviewText', '')}".strip()
    tokens = preprocess(raw_txt)

    item = {
        "review_id": rid,
        "userId": uid,
        "originalText": rv.get("reviewText", ""),
        "cleanedText": json.dumps(tokens),
        "containsProfanity": is_profane(raw_txt),
        "sentiment": _sentiment(raw_txt),
    }
    if rv.get("overall") is not None:
        item["overall"] = Decimal(str(rv["overall"]))

    try:  # ignore duplicate uploads
        reviews_tbl.put_item(
            Item=item, ConditionExpression="attribute_not_exists(review_id)"
        )
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise

    return {"status": "OK", "review_id": rid}
