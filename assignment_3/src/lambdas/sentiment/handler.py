"""
Lambda #3 – Sentiment analysis
Trigger : DynamoDB Streams (INSERT on Reviews)
Outcome : sets categorical sentiment (VADER only)
"""

import os, json, boto3, nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from pathlib import Path
from boto3.dynamodb.types import TypeDeserializer
deserializer = TypeDeserializer()


#ENDPOINT      = os.getenv("ENDPOINT", "http://localhost:4566")
REVIEWS_TABLE = os.getenv("REVIEWS_TABLE", "Reviews")

# bundled vader lexicon under nltk_data
NLTK_DIR = Path(__file__).parent / "nltk_data"
nltk.data.path.insert(0, str(NLTK_DIR))

sid = SentimentIntensityAnalyzer()
#ddb = boto3.resource("dynamodb", endpoint_url=ENDPOINT)
ddb = boto3.resource("dynamodb")
tbl = ddb.Table(REVIEWS_TABLE)

def handler(event, _ctx):
    for rec in event["Records"]:
        if rec["eventName"] != "INSERT":
            continue

        new_raw = rec["dynamodb"]["NewImage"]
        new = {k: deserializer.deserialize(v) for k, v in new_raw.items()}

        rid  = new["review_id"]
        toks = new["cleanedText"]
        txt  = " ".join(toks)

        score = sid.polarity_scores(txt)["compound"]
        label = (
            "positive" if score > 0.25
            else "negative" if score < -0.25
            else "neutral"
        )

        tbl.update_item(
            Key={"review_id": rid},
            UpdateExpression="SET sentiment = :s",
            ExpressionAttributeValues={":s": label},
        )
    