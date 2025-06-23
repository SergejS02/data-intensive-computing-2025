"""
End-to-end test:
 • upload a review to S3 ➜ preprocess Λ writes to Reviews table
 • profanity Λ marks containsProfanity
 • sentiment Λ adds label
 • after 4th profane review the user is banned
"""
import json, time, pytest
from conftest import wait_for_item

BUCKET  = "reviews-input"
REV_TAB = "Reviews"
USR_TAB = "Users"

# -------------------------------------------------------------  happy path
def test_pipeline_happy(boto, rnd_review):
    s3   = boto["s3"]
    ddb  = boto["dynamodb"]
    revT = ddb.Table(REV_TAB)

    # 1️⃣ upload review file (triggers preprocess Lambda)
    key = f"{rnd_review['review_id']}.json"
    s3.put_object(Bucket=BUCKET, Key=key,
                  Body=json.dumps(rnd_review).encode())

    # 2️⃣ wait until pipeline finishes
    item = wait_for_item(revT, {"review_id": rnd_review["review_id"]})

    # 3️⃣ asserts
    assert item["cleanedText"]                          # filled by preprocess
    assert item["containsProfanity"] is True            # from profanity Λ
    assert item["sentiment"] == "negative"              # from sentiment Λ

# -------------------------------------------------------------  ban logic
def test_user_gets_banned(boto, rnd_review):
    s3   = boto["s3"]
    ddb  = boto["dynamodb"]
    user = rnd_review["userId"]
    usrT = ddb.Table(USR_TAB)

    # Upload 4 profane reviews for same user
    for i in range(4):
        review = rnd_review | {"review_id": f"{rnd_review['review_id']}_{i}"}
        s3.put_object(Bucket=BUCKET,
                      Key=f"{review['review_id']}.json",
                      Body=json.dumps(review).encode())

    # Wait for user item
    user_item = wait_for_item(usrT, {"userId": user})

    assert user_item["unpoliteCnt"] >= 4
    assert user_item.get("banned") is True
