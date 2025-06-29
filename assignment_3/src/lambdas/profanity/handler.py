"""
Lambda #2 – Profanity check
Trigger : DynamoDB Streams (INSERT on Reviews)
Effects : sets containsProfanity, bumps unpoliteCnt, bans after >3
"""
import boto3, json, os
from profanityfilter import ProfanityFilter
from boto3.dynamodb.types import TypeDeserializer

deserializer = TypeDeserializer()

REVIEWS_TABLE = os.getenv("REVIEWS_TABLE", "Reviews")
USERS_TABLE   = os.getenv("USERS_TABLE", "Users")

ddb     = boto3.resource("dynamodb")
reviews = ddb.Table(REVIEWS_TABLE)
users   = ddb.Table(USERS_TABLE)

pf = ProfanityFilter()
CUSTOM_BAD = {"trash", "crap", "jerk"}

def handler(event, _ctx):
    for rec in event["Records"]:
        if rec["eventName"] != "INSERT":
            continue

        new_raw = rec["dynamodb"]["NewImage"]
        new = {k: deserializer.deserialize(v) for k, v in new_raw.items()}

        rid = new["review_id"]
        uid = new["userId"]
        text = new.get("originalText", "").lower()
        prof = pf.is_profane(text) or any(word in text for word in CUSTOM_BAD)

        reviews.update_item(
            Key={"review_id": rid},
            UpdateExpression="SET containsProfanity = :p",
            ExpressionAttributeValues={":p": prof},
        )

        if not prof:
            continue

        out = users.update_item(
            Key={"userId": uid},
            UpdateExpression="ADD unpoliteCnt :one",
            ExpressionAttributeValues={":one": 1},
            ReturnValues="UPDATED_NEW",
        )["Attributes"]

        if out.get("unpoliteCnt", 0) > 3:
            users.update_item(
                Key={"userId": uid},
                UpdateExpression="SET banned = :t",
                ExpressionAttributeValues={":t": True},
            )
