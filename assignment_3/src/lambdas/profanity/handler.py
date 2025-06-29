"""
Lambda #2 – profanity check + user‑strike counter
Trigger: NEW_IMAGE events from the *Reviews* DynamoDB stream.

• Overwrites `containsProfanity`.
• Increments `Users.unpoliteCnt`; sets `banned` once ≥ 4.

Exports `pf.is_profane(text)` for unit‑tests.
"""
from __future__ import annotations

import json, os, re
from decimal import Decimal
from types import SimpleNamespace
from urllib.parse import urlparse

import boto3

# ── helpers ─────────────────────────────────────────────────────────

def _ls_endpoint(port: int = 4566) -> str:
    ep = os.getenv("ENDPOINT")
    if ep:
        p = urlparse(ep)
        if p.hostname in ("localhost", "127.0.0.1"):
            host = os.getenv("LOCALSTACK_HOSTNAME", p.hostname)
            return f"{p.scheme}://{host}:{p.port or port}"
        return ep
    return f"http://{os.getenv('LOCALSTACK_HOSTNAME', 'localhost')}:{port}"

ENDPOINT = _ls_endpoint(); REGION = "us-east-1"
aws = dict(endpoint_url=ENDPOINT, region_name=REGION)

ddb = boto3.resource("dynamodb", **aws)
reviews_tbl = ddb.Table(os.getenv("REVIEWS_TABLE", "Reviews"))
users_tbl   = ddb.Table(os.getenv("USERS_TABLE",   "Users"))

# ── profanity heuristics (shared with tests) ─────────────────────────
TOK  = re.compile(r"[A-Za-z]+")
_BAD = {"shit", "fuck", "crap", "trash", "jerk", "horrible", "awful", "terrible"}

def _is_profane(text_or_tokens: str | list[str]) -> bool:
    """Return True if *text_or_tokens* contains any bad word."""
    if isinstance(text_or_tokens, str):
        tokens = TOK.findall(text_or_tokens.lower())
    else:
        tokens = text_or_tokens
    return any(t in _BAD for t in tokens)

# object imported by tests: pf.is_profane("text")
pf = SimpleNamespace(is_profane=_is_profane)

# ── handler ──────────────────────────────────────────────────────────

def handler(event, _ctx):
    handled = 0
    for rec in event["Records"]:
        if rec["eventName"] != "INSERT":
            continue

        img  = rec["dynamodb"]["NewImage"]
        rid  = img["review_id"]["S"]
        uid  = img["userId"]["S"]
        toks = json.loads(img["cleanedText"]["S"])

        prof = _is_profane(toks)

        reviews_tbl.update_item(
            Key={"review_id": rid},
            UpdateExpression="SET containsProfanity = :p",
            ExpressionAttributeValues={":p": prof},
        )

        if prof:
            resp = users_tbl.update_item(
                Key={"userId": uid},
                UpdateExpression="SET unpoliteCnt = if_not_exists(unpoliteCnt, :z) + :one",
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
