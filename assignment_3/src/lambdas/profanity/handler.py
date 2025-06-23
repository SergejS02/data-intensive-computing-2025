"""
Lambda #2 – Profanity check
Trigger : DynamoDB Streams (INSERT on Reviews)
Effects : sets containsProfanity, bumps unpoliteCnt, bans after >3
"""

import os, boto3, json
from profanityfilter import ProfanityFilter

ENDPOINT      = os.getenv("ENDPOINT", "http://localhost:4566")
REVIEWS_TABLE = os.getenv("REVIEWS_TABLE", "Reviews")
USERS_TABLE   = os.getenv("USERS_TABLE",   "Users")

ddb     = boto3.resource("dynamodb", endpoint_url=ENDPOINT)
reviews = ddb.Table(REVIEWS_TABLE)
users   = ddb.Table(USERS_TABLE)

pf = ProfanityFilter()
CUSTOM_BAD = {"trash","crap","jerk"}

def handler(event, _ctx):
    for rec in event["Records"]:
        if rec["eventName"]!="INSERT": continue
        new = rec["dynamodb"]["NewImage"]
        rid = new["review_id"]["S"]
        uid = new["userId"]["S"]
        toks= json.loads(new["cleanedText"]["S"])

        prof = pf.is_profane(" ".join(toks)) or any(tok in CUSTOM_BAD for tok in toks)
        reviews.update_item(
            Key={"review_id":rid},
            UpdateExpression="SET containsProfanity = :p",
            ExpressionAttributeValues={":p": prof},
        )

        if not prof: 
            continue
        # bump & check ban
        out = users.update_item(
            Key={"userId":uid},
            UpdateExpression="ADD unpoliteCnt :one",
            ExpressionAttributeValues={":one":1},
            ReturnValues="UPDATED_NEW",
        )["Attributes"]
        if out.get("unpoliteCnt",0)>3:
            users.update_item(
                Key={"userId":uid},
                UpdateExpression="SET banned = :t",
                ExpressionAttributeValues={":t":True},
            )
