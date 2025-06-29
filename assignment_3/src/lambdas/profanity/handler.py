"""
Lambda #2 – profanity detection & user banning.

Triggered by DynamoDB Streams (NEW_IMAGE) on the *Reviews* table.
Adds `containsProfanity` to the review item and increments the user's
`unpoliteCnt`.  The user is set `banned = true` after ≥ 4 profane reviews.
"""

from __future__ import annotations

import json
import os
from decimal import Decimal

import boto3

# ────────────────────────────── helpers ───────────────────────────────────
from urllib.parse import urlparse

def _ls_endpoint(port: int = 4566) -> str:
    """
    Resolve the correct LocalStack endpoint inside Lambda.

    • If ENDPOINT is set and NOT localhost → trust it (unit-tests).
    • If ENDPOINT is localhost → swap host for LOCALSTACK_HOSTNAME.
    • Otherwise derive from LOCALSTACK_HOSTNAME or default to localhost.
    """
    if (ep := os.getenv("ENDPOINT")):
        p = urlparse(ep)
        if p.hostname in ("localhost", "127.0.0.1"):
            host = os.getenv("LOCALSTACK_HOSTNAME", p.hostname)
            return f"{p.scheme}://{host}:{p.port or port}"
        return ep

    host = os.getenv("LOCALSTACK_HOSTNAME", "localhost")
    return f"http://{host}:{port}"



# ────────────────── profanity filter (with stub) ──────────────────────────
try:
    from profanityfilter import ProfanityFilter
except ModuleNotFoundError:
    class ProfanityFilter:                               # minimal stub
        def is_profane(self, _text: str) -> bool:
            return False

BAD_WORDS = {
    "shit", "fuck", "crap", "trash", "jerk",
    "horrible", "awful", "terrible",
}


def _is_profane(text: str) -> bool:
    base  = _pf.is_profane(text)
    extra = any(t in BAD_WORDS for t in text.lower().split())
    return base or extra


class _Proxy:
    def is_profane(self, txt: str) -> bool:
        return _is_profane(txt)


_pf = ProfanityFilter()
pf  = _Proxy()      # what unit tests import


# ─────────────────────────── AWS clients ──────────────────────────────────
ENDPOINT = _ls_endpoint()
REGION   = "us-east-1"

ddb = boto3.resource("dynamodb", endpoint_url=ENDPOINT, region_name=REGION)
REVIEWS_TABLE = os.getenv("REVIEWS_TABLE", "Reviews")
USERS_TABLE   = os.getenv("USERS_TABLE",   "Users")
reviews_tbl   = ddb.Table(REVIEWS_TABLE)
users_tbl     = ddb.Table(USERS_TABLE)


# ───────────────────────────── handler ────────────────────────────────────
def handler(event, _ctx):
    for rec in event["Records"]:
        if rec["eventName"] != "INSERT":
            continue

        img  = rec["dynamodb"]["NewImage"]
        rid  = img["review_id"]["S"]
        uid  = img["userId"]["S"]
        toks = json.loads(img["cleanedText"]["S"])

        flag = _is_profane(" ".join(toks))

        # 1) tag the review
        reviews_tbl.update_item(
            Key={"review_id": rid},
            UpdateExpression="SET containsProfanity = :p",
            ExpressionAttributeValues={":p": flag},
        )

        # 2) increment user's counter & possibly ban
        resp = users_tbl.update_item(
            Key={"userId": uid},
            UpdateExpression="""
                SET unpoliteCnt = if_not_exists(unpoliteCnt, :z) + :one
            """,
            ExpressionAttributeValues={
                ":z": Decimal(0),
                ":one": Decimal(1),
            },
            ReturnValues="UPDATED_NEW",
        )
        strikes = int(resp["Attributes"]["unpoliteCnt"])
        if strikes >= 4:
            users_tbl.update_item(
                Key={"userId": uid},
                UpdateExpression="SET banned = :b",
                ExpressionAttributeValues={":b": True},
            )

    return {"processed": len(event["Records"])}
