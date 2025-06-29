"""
Lambda #1 – Pre-processing
Trigger : (driven externally) JSON payloads
Output  : Normalised review → DynamoDB “Reviews” table
Helper  : preprocess(text) → list[str]  (imported by unit tests)
"""

import os, json, uuid, re, boto3, nltk
from decimal import Decimal
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer

# config via ENV
#ENDPOINT      = os.getenv("ENDPOINT", "http://localhost:4566")
REVIEWS_TABLE = os.getenv("REVIEWS_TABLE", "Reviews")

# AWS clients
#s3  = boto3.client("s3",      endpoint_url=ENDPOINT)
#ddb = boto3.resource("dynamodb", endpoint_url=ENDPOINT)
s3 = boto3.client("s3")
ddb = boto3.resource("dynamodb")

# NLTK data (bundled under src/lambdas/preprocess/nltk_data)
NLTK_DIR = Path(__file__).parent / "nltk_data"
nltk.data.path.insert(0, str(NLTK_DIR))

TOKEN_RE   = re.compile(r"\b[A-Za-z]+\b")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text: str) -> list[str]:
    return [
        lemmatizer.lemmatize(tok)
        for tok in TOKEN_RE.findall(text.lower())
        if tok not in stop_words
    ]

def handler(event, _ctx):
    """
    Handle S3 trigger event. Download JSON from S3, parse it, preprocess it,
    and insert into the DynamoDB Reviews table.
    """

    for record in event.get("Records", []):
        s3_info = record.get("s3", {})
        bucket = s3_info["bucket"]["name"]
        key = s3_info["object"]["key"]

        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj['Body'].read()
        review = json.loads(body)

        uid = review.get("userId")
        review_id = review.get("review_id")
        text = f"{review.get('summary','')} {review.get('reviewText','')}".strip()
        cleaned = preprocess(text)

        tbl = ddb.Table(REVIEWS_TABLE)
        tbl.put_item(Item={
            "review_id":         review_id,
            "userId":            uid,
            "overall":           Decimal(str(review["overall"])),
            "originalText":      review.get("reviewText",""),
            "cleanedText":       cleaned,
            "containsProfanity": None,
            "sentiment":         None,
        })

        print(f"Inserted review {review_id} for user {uid} into DynamoDB")

    return {"status": "PREPROCESSED"}
