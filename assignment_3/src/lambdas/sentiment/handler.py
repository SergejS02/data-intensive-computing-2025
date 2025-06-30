from __future__ import annotations
import json
import os
import boto3
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from urllib.parse import urlparse

def _ls_endpoint(port: int = 4566) -> str:
    if (ep := os.getenv("ENDPOINT")):
        p = urlparse(ep)
        if p.hostname in ("localhost", "127.0.0.1"):
            host = os.getenv("LOCALSTACK_HOSTNAME", p.hostname)
            return f"{p.scheme}://{host}:{p.port or port}"
        return ep

    host = os.getenv("LOCALSTACK_HOSTNAME", "localhost")
    return f"http://{host}:{port}"


#nltk vader computes score with threshold >0.05 and <-0.05
NLTK_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.insert(0, NLTK_DIR)

sid = SentimentIntensityAnalyzer()         
_ANALYSER = sid                            


def _label(text: str) -> str:
    score = sid.polarity_scores(text)["compound"]
    if score > 0.05:
        return "positive"
    if score < -0.05:
        return "negative"
    return "neutral"


ENDPOINT = _ls_endpoint()
REGION   = "us-east-1"

ddb = boto3.resource("dynamodb", endpoint_url=ENDPOINT, region_name=REGION)
reviews_tbl = ddb.Table(os.getenv("REVIEWS_TABLE", "Reviews"))


def handler(event, _ctx):
    for rec in event["Records"]:
        if rec["eventName"] != "INSERT":
            continue

        img  = rec["dynamodb"]["NewImage"]
        rid  = img["review_id"]["S"]
        toks = json.loads(img["cleanedText"]["S"])

        reviews_tbl.update_item(
            Key={"review_id": rid},
            UpdateExpression="SET sentiment = :s",
            ExpressionAttributeValues={":s": _label(' '.join(toks))},
        )

    return {"processed": len(event["Records"])}
