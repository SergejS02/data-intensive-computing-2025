#!/usr/bin/env python
"""
bulk_loader.py  –  blast a huge review file into S3 as fast as possible.

Usage examples
--------------
# 32-way concurrency, gzip file
python tools/bulk_loader.py reviews_devset.json.gz 32

# default 16 threads, plain JSON-lines file
python tools/bulk_loader.py reviews_10k.jsonl
"""
from __future__ import annotations
import json, gzip, os, sys, uuid, concurrent.futures, pathlib, boto3, tqdm

# ── CLI args ─────────────────────────────────────────────────────────────
FILE     = pathlib.Path(sys.argv[1])
WORKERS  = int(sys.argv[2]) if len(sys.argv) > 2 else 16
BUCKET   = os.getenv("INPUT_BUCKET", "reviews-input")
ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")

# ── AWS client (works for LocalStack *and* real AWS) ─────────────────────
s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)

# ── upload helper ────────────────────────────────────────────────────────
def put(line: str) -> None:
    review = json.loads(line)
    key    = f"{review.get('review_id') or uuid.uuid4()}.json"
    body   = json.dumps(review).encode()
    s3.put_object(Bucket=BUCKET, Key=key, Body=body)

open_fn = gzip.open if FILE.suffix == ".gz" else open
with open_fn(FILE, "rt") as fh, \
     concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
    list(tqdm.tqdm(ex.map(put, fh, chunksize=256), desc="Uploading"))

print(f"Queued all reviews from {FILE} with {WORKERS} workers.")
print("Letting the pipeline drain for 600 s …")
import time; time.sleep(600)
print("✓ Done – now run  python print_results.py")
