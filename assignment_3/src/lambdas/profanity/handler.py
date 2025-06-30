from __future__ import annotations

"""
Lambda #2 – profanity check + strike counter.

Trigger: NEW_IMAGE on the **Reviews** DynamoDB stream.

• Re‑computes `containsProfanity` using the assignment’s eight‑word list **plus**
  the lexicon from the **`profanityfilter`** package (import name has **no dash**).
• Keeps a per‑user strike counter in a shadow item `_tmp_<userId>`; at 4 strikes
  it writes a real Users row with `banned=True` (idempotent) and prunes the
  shadow key.

The handler now *unconditionally* imports `profanityfilter.ProfanityFilter` –
there is no fallback to the dash version.
"""

import json
import os
import re
from decimal import Decimal
from types import SimpleNamespace
from urllib.parse import urlparse

import boto3
import botocore.exceptions
from profanityfilter import ProfanityFilter  # ← always this library

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

# ── profanity logic ─────────────────────────────────────────────────────

_pf_lib = ProfanityFilter()
# short‑circuit the heavy censor function to a constant so .is_profane() stays cheap
if hasattr(_pf_lib, "set_censor_func"):
    _pf_lib.set_censor_func(lambda _: True)

TOK = re.compile(r"[A-Za-z]+")
_CUSTOM_BAD = {
    "shit",
    "fuck",
    "crap",
    "trash",
    "jerk",
    "horrible",
    "awful",
    "terrible",
}

def _is_profane(arg: str | list[str]) -> bool:
    """Return True if the review trips either the custom list or the library."""
    text = " ".join(arg) if isinstance(arg, list) else arg
    toks = TOK.findall(text.lower())
    return any(t in _CUSTOM_BAD for t in toks) or _pf_lib.is_profane(text)

# object exposed for unit‑tests
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

        # update profanity flag in Reviews row
        reviews_tbl.update_item(
            Key={"review_id": rid},
            UpdateExpression="SET containsProfanity = :p",
            ExpressionAttributeValues={":p": prof},
        )

        if not prof:
            continue  # polite review – no strike

        # ── strike counting (shadow item) ────────────────────────────────
        tmp_key = f"_tmp_{uid}"
        resp = users_tbl.update_item(
            Key={"userId": tmp_key},
            UpdateExpression="ADD cnt :one",
            ExpressionAttributeValues={":one": Decimal(1)},
            ReturnValues="UPDATED_NEW",
        )
        strikes = int(resp["Attributes"]["cnt"])

        if strikes >= 4:
            # promote to real user row (idempotent)
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
                if e.response["Error"]["Code"] != "ConditionalCheckFailedException":
                    raise
            # tidy up shadow counter
            users_tbl.delete_item(Key={"userId": tmp_key})

    return {"handled": len(event["Records"])}
