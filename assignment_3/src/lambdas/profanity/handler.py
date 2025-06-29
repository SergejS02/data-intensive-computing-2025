"""
Lambda #2 – profanity check + (user) strike counter.

Trigger: NEW_IMAGE records from the Reviews table stream.

• Re-computes containsProfanity (idempotent overwrite).
• Keeps per-user strike count in a *shadow* item with key "_tmp_<userId>".
  Once strikes ≥ 4 → creates the real Users row with `unpoliteCnt` and `banned`.
• Exports `pf.is_profane(text)` for unit tests.
"""
from __future__ import annotations

import json
import os
import re
from decimal import Decimal
from types import SimpleNamespace
from urllib.parse import urlparse

import boto3
import botocore.exceptions

# ── AWS helpers ──────────────────────────────────────────────────────────
def _ls_endpoint(port: int = 4566) -> str:
    ep = os.getenv("ENDPOINT")
    if ep:
        p = urlparse(ep)
        if p.hostname in ("localhost", "127.0.0.1"):
            host = os.getenv("LOCALSTACK_HOSTNAME", p.hostname)
            return f"{p.scheme}://{host}:{p.port or port}"
        return ep
    return f"http://{os.getenv('LOCALSTACK_HOSTNAME', 'localhost')}:{port}"


aws_cfg = dict(endpoint_url=_ls_endpoint(), region_name="us-east-1")
ddb = boto3.resource("dynamodb", **aws_cfg)
reviews_tbl = ddb.Table(os.getenv("REVIEWS_TABLE", "Reviews"))
users_tbl = ddb.Table(os.getenv("USERS_TABLE", "Users"))

# ── profanity logic (shared with tests) ──────────────────────────────────
TOK = re.compile(r"[A-Za-z]+")
_BAD = {"shit", "fuck", "crap", "trash", "jerk", "horrible", "awful", "terrible"}


def _is_profane(arg: str | list[str]) -> bool:
    toks = TOK.findall(arg.lower()) if isinstance(arg, str) else arg
    return any(t in _BAD for t in toks)


pf = SimpleNamespace(is_profane=_is_profane)

# ── Lambda handler ───────────────────────────────────────────────────────
def handler(event, _ctx):
    for rec in event["Records"]:
        if rec["eventName"] != "INSERT":
            continue

        img = rec["dynamodb"]["NewImage"]
        rid = img["review_id"]["S"]
        uid = img["userId"]["S"]
        toks = json.loads(img["cleanedText"]["S"])

        prof = _is_profane(toks)

        reviews_tbl.update_item(
            Key={"review_id": rid},
            UpdateExpression="SET containsProfanity = :p",
            ExpressionAttributeValues={":p": prof},
        )

        if not prof:  # polite review – nothing to do
            continue

        # ---- strike counting (shadow item) ---------------------------------
        tmp_key = f"_tmp_{uid}"
        resp = users_tbl.update_item(
            Key={"userId": tmp_key},
            UpdateExpression="ADD cnt :one",
            ExpressionAttributeValues={":one": Decimal(1)},
            ReturnValues="UPDATED_NEW",
        )
        strikes = int(resp["Attributes"]["cnt"])

        if strikes >= 4:
            # promote to real user row (once)
            try:
                users_tbl.put_item(
                    Item={
                        "userId": uid,
                        "unpoliteCnt": Decimal(strikes),
                        "banned": True,
                    },
                    ConditionExpression="attribute_not_exists(userId)",
                )
            except botocore.exceptions.ClientError as e:
                if (
                    e.response["Error"]["Code"]
                    != "ConditionalCheckFailedException"
                ):
                    raise
            # best-effort cleanup of the shadow counter
            users_tbl.delete_item(Key={"userId": tmp_key})

    return {"handled": len(event["Records"])}
