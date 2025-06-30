import os, time, subprocess, json, pathlib, sys, uuid, random
os.environ.setdefault("AWS_DEFAULT_REGION",        "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID",         "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY",     "test")
os.environ.setdefault("AWS_ENDPOINT_URL",          "http://localhost:4566")
import pytest, boto3

#src importable
ROOT = pathlib.Path(__file__).resolve().parents[1]  
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))                        

#aws connection 
ENDPOINT = os.getenv("AWS_ENDPOINT", "http://localhost:4566")
AWS_OPTS = dict(endpoint_url=ENDPOINT, region_name="us-east-1")

#CLI option
def pytest_addoption(parser):
    parser.addoption(
        "--sample",
        type=int,
        metavar="N",
        help="Run the dev-set integration test on a random sample of N reviews "
             "(omit the flag for full dataset).",
    )


@pytest.fixture(scope="session")
def boto():
    """Return a dict of boto3 clients/resources pre-wired to LocalStack."""
    return {
        "s3"      : boto3.client("s3", **AWS_OPTS),
        "dynamodb": boto3.resource("dynamodb", **AWS_OPTS),
    }

@pytest.fixture(scope="function")
def rnd_review():
    """Return a fresh review dict each time."""
    uid = f"user{random.randint(1000, 9999)}"
    return {
        "review_id"  : str(uuid.uuid4()),
        "userId"     : uid,
        "reviewText" : "horrible trash product",
        "summary"    : "awful",
        "overall"    : 1,
    }


def wait_for_item(table, key, timeout=600):
    """Poll DynamoDB until the item exists or timeout (seconds)."""
    for _ in range(timeout):
        resp = table.get_item(Key=key)
        if "Item" in resp:
            return resp["Item"]
        time.sleep(0.1)
    raise TimeoutError("Item not found in DynamoDB within timeout")
