"""
End-to-end test:
 • upload a review to S3 ➜ preprocess Λ writes to Reviews table
 • profanity Λ marks containsProfanity
 • sentiment Λ adds label
 • after 4th profane review the user is banned
"""
import json, time, uuid, random, decimal

BUCKET  = "reviews-input"
REV_TAB = "Reviews"
USR_TAB = "Users"

# -------------------------------------------------------------  helpers

def run_pipeline(boto, review, timeout=60):
    """Upload a review to S3 and wait for it to be processed."""
    s3   = boto["s3"]
    ddb  = boto["dynamodb"]
    revT = ddb.Table(REV_TAB)
    s3_object_key = f"{review['review_id']}.json"
    review_db_key = {"review_id": review["review_id"]}

    s3.put_object(Bucket=BUCKET, Key=s3_object_key,
                  Body=json.dumps(review).encode())

    item = None
    for _ in range(timeout):
        resp = revT.get_item(Key=review_db_key)
        if "Item" in resp and resp["Item"].get("sentiment") is not None and resp["Item"].get("containsProfanity") is not None:
            item = resp["Item"]
            break
        else:
            time.sleep(1)
    if item is not None:
        return item
    else:
        raise TimeoutError(f"Item {review_db_key} not found in DynamoDB within timeout")
# -------------------------------------------------------------  test cases

def test_good_review(boto):
    """Test a good review with no profanity."""
    good_review = {
        "review_id": f"good-{uuid.uuid4().hex[:4]}",
        "userId": "user-good",
        "summary": "Nice",
        "reviewText": "I love it. Amazing product!",
        "overall": 5
    }
    item = run_pipeline(boto,good_review)
    assert item["sentiment"] == "positive"
    assert item["containsProfanity"] is False

def test_bad_review(boto):
    """Test a bad review with profanity."""
    bad_review = {
            "review_id": f"bad-{uuid.uuid4().hex[:4]}",
            "userId": "user-bad",
            "summary": "Awful",
            "reviewText": "This is total crap.",
            "overall": 1
            }
    item = run_pipeline(boto, bad_review)
    assert item["sentiment"] == "negative"
    assert item["containsProfanity"] is True

def test_edge_case_review(boto):
    """Test an edge case review with no summary or text."""
    review = {
            "review_id": f"edge-{uuid.uuid4().hex[:4]}",
            "userId": "user-edge",
            "summary": "",
            "reviewText": "",
            "overall": 3
        }
    item = run_pipeline(boto, review)
    assert item["sentiment"] not in ("positive", "negative")
    assert item["containsProfanity"] is False

# -------------------------------------------------------------  ban logic
def test_user_gets_banned(boto):
    profane_reviews_to_ban = 4
    
    s3   = boto["s3"]
    ddb  = boto["dynamodb"]
    usrT = ddb.Table(USR_TAB)
    revT = ddb.Table(REV_TAB)

    user_id = f"test-user-{uuid.uuid4().hex[:4]}"
    base_review = {
        "userId": user_id,
        "summary": "bad",
        "reviewText": "this is trash",
        "overall": 1
    }

    for i in range(profane_reviews_to_ban):
        # customize each review slightly to ensure uniqueness
        review = base_review.copy()
        review["review_id"] = f"{user_id}-review-{i}"
        review["reviewText"] = f"Review {i+1}: {base_review['reviewText']} - {random.choice(['crap', 'damn', 'ass', 'hell'])} words added."

        # upload
        s3_object_key = f"{review['review_id']}.json"
        review_db_key = {"review_id": review["review_id"]}
        s3.put_object(Bucket=BUCKET, Key=s3_object_key,
                      Body=json.dumps(review).encode())
        
        # wait to be processed in the Reviews Table
        processed_review_item = None
        for _ in range(60): # timeout 60s
            resp = revT.get_item(Key=review_db_key)
            if "Item" in resp and \
               resp["Item"].get("sentiment") is not None and \
               resp["Item"].get("containsProfanity") is not None:
                processed_review_item = resp["Item"]
                break
            time.sleep(1)

        if processed_review_item is None:
            raise TimeoutError(f"Review {review_db_key} not fully processed in Reviews table.")

        assert processed_review_item["containsProfanity"] is True, \
            f"Review {review['review_id']} was expected to contain profanity but did not."
        assert processed_review_item["sentiment"] == "negative", \
            f"Review {review['review_id']} sentiment was expected to be negative, but was {processed_review_item['sentiment']}."

    # check the Users table
    user_item_key = {"userId": user_id}
    final_user_item = None
    for _ in range(120): # timeout 120s
        resp = usrT.get_item(Key=user_item_key)
        if "Item" in resp and \
           resp["Item"].get("banned") is True and \
           resp["Item"].get("unpoliteCnt") is not None and \
           int(resp["Item"].get("unpoliteCnt")) >= profane_reviews_to_ban:
            final_user_item = resp["Item"]
            break
        time.sleep(1)

    if final_user_item is None:
        current_user_state = usrT.get_item(Key=user_item_key).get("Item", {})
        raise TimeoutError(
            f"User {user_item_key} not banned or unpoliteCnt not updated to >= {profane_reviews_to_ban}. "
            f"Current user state: {current_user_state}"
        )

    # Final assertions on the user item
    assert "unpoliteCnt" in final_user_item, "unpoliteCnt not found in user item"
    # DynamoDB numbers are Decimal, convert to int for comparison
    assert isinstance(final_user_item["unpoliteCnt"], (int, float, decimal.Decimal)), \
           f"unpoliteCnt is not a number, got {type(final_user_item['unpoliteCnt'])}"
    assert int(final_user_item["unpoliteCnt"]) >= profane_reviews_to_ban, \
        f"Expected unpoliteCnt >= {profane_reviews_to_ban}, got {int(final_user_item['unpoliteCnt'])}"
    
    assert "banned" in final_user_item, "banned status not found in user item"
    assert final_user_item["banned"] is True, \
        f"User {user_id} was not banned. Banned status: {final_user_item['banned']}"
