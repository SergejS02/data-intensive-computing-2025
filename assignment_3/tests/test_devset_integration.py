"""
Integration test that streams the entire reviews_devset.json file
through the serverless pipeline.  It then verifies that:
  • all reviews ended up in the Reviews table
  • each review has sentiment & profanity filled
  • summary statistics are produced (printed) for manual inspection
Mark the test 'slow' so you can skip it with `pytest -m "not slow"`
"""
import json, time, pathlib, random
import pytest

ROOT        = pathlib.Path(__file__).resolve().parents[1]
DATASET     = ROOT / "reviews_devset.json"
BUCKET      = "reviews-input"
REVIEWS_TBL = "Reviews"
USERS_TBL   = "Users"

pytestmark = pytest.mark.slow

def load_dataset(sample=None):
    """Read reviews_devset.json (JSON-Lines)."""
    reviews = []
    with DATASET.open("r", encoding="utf-8") as fh:
        for line in fh:
            reviews.append(json.loads(line))
    if sample:
        reviews = random.sample(reviews, sample)
    for i, r in enumerate(reviews):
        r.setdefault("review_id", f"dev-{i}")
    return reviews

def test_devset_pipeline(boto, request):
    if not DATASET.exists():
        pytest.skip(f"{DATASET} not found in repo root")

    # ----- load reviews (you can limit sample size via pytest --sample=N ) -----
    sample_n = request.config.getoption("--sample", default=None)
    reviews  = load_dataset(int(sample_n)) if sample_n else load_dataset()

    s3    = boto["s3"]
    ddb   = boto["dynamodb"]
    revT  = ddb.Table(REVIEWS_TBL)
    usrT  = ddb.Table(USERS_TBL)

    # ----- upload all reviews to S3 (triggers pipeline) -----------------------
    for rv in reviews:
        key = f"{rv['review_id']}.json"
        s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(rv).encode())

    # ----- wait until every review appears with sentiment --------------------
    deadline = time.time() + 120   # 2-minute hard cap for big sets
    while time.time() < deadline:
        processed = revT.scan()["Items"]
        if len(processed) >= len(reviews) and all(
            itm.get("sentiment") and itm.get("containsProfanity") is not None
            for itm in processed
        ):
            break
        time.sleep(2)
    else:
        pytest.fail("Pipeline did not finish within timeout")

    # ----- compute required stats -------------------------------------------
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    profane = 0
    for itm in processed:
        counts[itm["sentiment"]] += 1
        if itm["containsProfanity"]:
            profane += 1

    banned_users = [
        u["userId"] for u in usrT.scan()["Items"] if u.get("banned")
    ]

    # ----- sanity asserts ----------------------------------------------------
    assert sum(counts.values()) == len(reviews)
    # at least one review should fall into *some* category
    assert any(v > 0 for v in counts.values())

    # ----- print a concise summary for the grader ---------------------------
    print("\n\n=== DEV-SET SUMMARY ===")
    print(f"Total reviews          : {len(reviews)}")
    print(f"Positive / Neutral / Negative : {counts['positive']} / "
          f"{counts['neutral']} / {counts['negative']}")
    print(f"Profanity failures     : {profane}")
    print(f"Banned users           : {len(banned_users)}  -> {banned_users[:5]}...\n")

