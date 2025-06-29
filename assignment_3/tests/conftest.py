import os
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")   # <─ fixes NoRegionError
import os, time, subprocess, json, pathlib, sys, uuid, random
os.environ.setdefault("AWS_DEFAULT_REGION",        "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID",         "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY",     "test")
#os.environ.setdefault("AWS_ENDPOINT_URL",          "http://localhost:4566")
import pytest, boto3

# ── make “src” importable ─────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]   # …/assignment_3
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))                         # ← key line
# ──────────────────────────────────────────────────────────────────────

# ------------------------------------------------------------------  AWS
ENDPOINT = os.getenv("AWS_ENDPOINT", "http://localhost:4566")
AWS_OPTS = dict(endpoint_url=ENDPOINT, region_name="us-east-1")

# ------------------------------------------------------------------  custom CLI option
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
