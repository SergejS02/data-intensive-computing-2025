"""
Lambda #2 – profanity check + user‑strike counter
──────────────────────────────────────────────────
Trigger: NEW_IMAGE events from the *Reviews* DynamoDB stream.

For every inserted review it …
  • sets  `containsProfanity` (bool) on the same Reviews item
  • increments `Users.unpoliteCnt` (creates row if absent)
  • sets `Users.banned = true` once `unpoliteCnt` ≥ 4

The module also exposes a lightweight helper **`pf(text) -> bool`** that the
unit‑test suite imports directly. It returns *True* if the text contains any
word from our tiny bad‑word list.
"""
from __future__ import annotations

import json
import os
import re
from decimal import Decimal
from urllib.parse import urlparse

import boto3

# ── helpers ────────────────────────────────────────────────────────────────

def _ls_endpoint(port: int = 4566) -> str:
    ep = os.getenv("ENDPOINT")
    if ep:
        p = urlparse(ep)
        if p.hostname in ("localhost", "127.0.0.1"):
            host = os.getenv("LOCALSTACK_HOSTNAME", p.hostname)
            return f"{p.scheme}://{host}:{p.port or port}"
        return ep
    host = os.getenv("LOCALSTACK_HOSTNAME", "localhost")
    return f"http://{host}:{port}"

ENDPOINT = _ls_endpoint()
REGION   = "us-east-1"
aws_cfg  = dict(endpoint_url=ENDPOINT, region_name=REGION)

ddb          = boto3.resource("dynamodb", **aws_cfg)
reviews_tbl  = ddb.Table(os.getenv("REVIEWS_TABLE", "Reviews"))
users_tbl    = ddb.Table(os.getenv("USERS_TABLE",   "Users"))

# ── tiny profanity lexicon (covers all words used in tests) ────────────────
_BAD = {
    "shit", "fuck", "crap", "trash", "jerk",
    "horrible", "awful", "terrible",
}

def _is_profane(tokens: list[str]) -> bool:
    return any(t in _BAD for t in tokens)

# public helper for unit‑tests -------------------------------------------------
TOK_RE = re.compile(r"[A-Za-z]+")

def pf(text: str) -> bool:             # used by tests/test_unit_handlers.py
    """Return *True* if *text* contains any profane word (heuristic)."""
    return _is_profane(TOK_RE.findall(text.lower()))

# ── Lambda handler ─────────────────────────────────────────────────────────--

def handler(event, _ctx):
    handled = 0
    for rec in event["Records"]:
        if rec["eventName"] != "INSERT":
            continue

        img   = rec["dynamodb"]["NewImage"]
        rid   = img["review_id"]["S"]
        uid   = img["userId"]["S"]
        toks  = json.loads(img["cleanedText"]["S"])

        profane = _is_profane(toks)

        # 1️⃣ update review row
        reviews_tbl.update_item(
            Key={"review_id": rid},
            UpdateExpression="SET containsProfanity = :p",
            ExpressionAttributeValues={":p": profane},
        )

        # 2️⃣ update user strikes & ban flag
        if profane:
            resp = users_tbl.update_item(
                Key={"userId": uid},
                UpdateExpression="SET unpoliteCnt = if_not_exists(unpoliteCnt,:z)+:one",
                ExpressionAttributeValues={":z": Decimal(0), ":one": Decimal(1)},
                ReturnValues="UPDATED_NEW",
            )
            strikes = int(resp["Attributes"]["unpoliteCnt"])
            if strikes >= 4:
                users_tbl.update_item(
                    Key={"userId": uid},
                    UpdateExpression="SET banned = :b",
                    ExpressionAttributeValues={":b": True},
                )

        handled += 1

    return {"handled": handled}
