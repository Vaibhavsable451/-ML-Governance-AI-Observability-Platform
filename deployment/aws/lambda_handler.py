"""Event-driven governance checks. Stdlib + boto3 only (boto3 ships with the Lambda runtime).

Triggers
  * S3 ObjectCreated on  models/<model_id>/...  (model or dataset uploaded)  -> governance_check
  * EventBridge schedule, e.g. rate(1 hour)  {"task": "governance_check"}    -> re-evaluate every model
  * EventBridge schedule                     {"task": "cost_check"}          -> FinOps budget alarm
Env: API_URL (http://<ec2-public-dns>:30081 or an ALB), SNS_TOPIC_ARN (optional), NAMESPACE (default AIGovernance)
"""
import json
import os
import urllib.request

import boto3

API_URL = os.environ["API_URL"].rstrip("/")
NAMESPACE = os.getenv("NAMESPACE", "AIGovernance")
cw = boto3.client("cloudwatch")


def _api(method, path):
    req = urllib.request.Request(API_URL + path, method=method, data=b"" if method == "POST" else None)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def _metric(name, value, unit="None", **dims):
    cw.put_metric_data(Namespace=NAMESPACE, MetricData=[{
        "MetricName": name, "Value": float(value), "Unit": unit,
        "Dimensions": [{"Name": k, "Value": v} for k, v in dims.items()]}])


def _notify(subject, body):
    arn = os.getenv("SNS_TOPIC_ARN")
    if arn:
        boto3.client("sns").publish(TopicArn=arn, Subject=subject[:100], Message=json.dumps(body, default=str))


def governance_check(model_id, scenario="none"):
    p = _api("POST", f"/models/{model_id}/evaluate?scenario={scenario}")
    _metric("LambdaRiskScore", p["risk"]["score"], ModelId=model_id)
    _metric("LambdaDriftScore", p["drift"]["drift_score"], ModelId=model_id)
    if p["risk"]["decision"] == "HUMAN_REVIEW":
        _notify(f"{model_id} needs review (risk {p['risk']['score']})",
                {"model": model_id, "level": p["risk"]["level"], "reasons": [r["text"] for r in p["risk"]["reasons"]]})
    return {"model": model_id, "risk": p["risk"]["score"], "status": p["status"]}


def cost_check():
    f = _api("GET", "/finops")
    pct = f["cost_usd"] / f["budget_usd"] * 100 if f["budget_usd"] else 0
    _metric("BudgetUsedPercent", pct, "Percent")
    if pct >= 80:
        _notify("LLM budget above 80%", f)
    return {"budget_used_pct": round(pct, 2)}


def handler(event, context):
    task = event.get("task", "governance_check")
    if "Records" in event:  # S3 notification: models/<model_id>/file
        ids = {r["s3"]["object"]["key"].split("/")[1] for r in event["Records"] if r["s3"]["object"]["key"].count("/") >= 2}
        out = [governance_check(i) for i in ids]
    elif task == "cost_check":
        out = cost_check()
    else:
        ids = [event["model_id"]] if event.get("model_id") else [m["model_id"] for m in _api("GET", "/models")]
        out = [governance_check(i, event.get("scenario", "none")) for i in ids]
    print(json.dumps({"task": task, "result": out}))
    return {"statusCode": 200, "body": json.dumps(out)}
