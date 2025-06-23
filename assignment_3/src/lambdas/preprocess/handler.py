"""
Lambda #1 – Pre-processing
Trigger : (driven externally) JSON payloads
Output  : Normalised review → DynamoDB “Reviews” table
Helper  : preprocess(text) → list[str]  (imported by unit tests)
"""

import os, json, uuid, re, boto3, nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer

# config via ENV
ENDPOINT      = os.getenv("ENDPOINT", "http://localhost:4566")
REVIEWS_TABLE = os.getenv("REVIEWS_TABLE", "Reviews")

# AWS clients
s3  = boto3.client("s3",      endpoint_url=ENDPOINT)
ddb = boto3.resource("dynamodb", endpoint_url=ENDPOINT)

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
    Expect a raw JSON review payload in event['body'] or encoded directly as event.
    For our watcher, we send the JSON itself as the entire event.
    """
    review = event if isinstance(event, dict) else json.loads(event)

    # fallback on reviewerID if userId missing
    uid = review.get("userId") or review.get("reviewerID") or "unknown"

    review_id = review.get("review_id") or str(uuid.uuid4())
    text      = f"{review.get('summary','')} {review.get('reviewText','')}".strip()
    cleaned   = preprocess(text)

    tbl = ddb.Table(REVIEWS_TABLE)
    tbl.put_item(Item={
        "review_id":         review_id,
        "userId":            uid,
        "overall":           review.get("overall"),
        "originalText":      review.get("reviewText",""),
        "cleanedText":       cleaned,
        "containsProfanity": None,
        "sentiment":         None,
    })
    return {"status":"PREPROCESSED","review_id":review_id}
