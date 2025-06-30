#!/usr/bin/env python

from collections import Counter
import os, boto3, tabulate

ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")
REGION   = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

AWS = dict(endpoint_url=ENDPOINT, region_name=REGION,
           aws_access_key_id="test", aws_secret_access_key="test")

dynamodb = boto3.resource("dynamodb", **AWS)

REVIEWS = dynamodb.Table("Reviews")
USERS   = dynamodb.Table("Users")

#fetch every item in reviews table
def scan_all(table):
    items, last = [], None
    while True:
        kwargs = {"ExclusiveStartKey": last} if last else {}
        resp   = table.scan(**kwargs)
        items.extend(resp["Items"])
        if "LastEvaluatedKey" not in resp:
            return items
        last = resp["LastEvaluatedKey"]

reviews = scan_all(REVIEWS)

#aggr
sent_counts = Counter(r["sentiment"] for r in reviews)
profane     = sum(1 for r in reviews if r.get("containsProfanity"))

users  = scan_all(USERS)
banned = [u["userId"] for u in users if u.get("banned")]

#show results
print("\nRESULTS for reviews_devset.json")

print(tabulate.tabulate([
    ["Positive", sent_counts["positive"]],
    ["Neutral",  sent_counts["neutral"]],
    ["Negative", sent_counts["negative"]],
    ["Profanity fails", profane],
], tablefmt="github"))

print(f"\nBanned users: {len(banned)}")
if banned:
    print("Sample:", ", ".join(banned[:10]), "...")

