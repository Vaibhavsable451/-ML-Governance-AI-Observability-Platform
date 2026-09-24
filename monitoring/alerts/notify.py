"""Alert fan-out: log always, SNS when SNS_TOPIC_ARN is configured."""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger("alerts")


def send_alert(subject: str, payload: dict) -> None:
    log.warning("ALERT %s", subject, extra={"extra_fields": {"alert": payload}})
    arn = os.getenv("SNS_TOPIC_ARN")
    if not arn:
        return
    try:
        import boto3
        boto3.client("sns", region_name=os.getenv("AWS_REGION", "us-east-1")).publish(
            TopicArn=arn, Subject=subject[:100], Message=json.dumps(payload, default=str))
    except Exception as exc:
        log.error("sns publish failed: %s", exc)
