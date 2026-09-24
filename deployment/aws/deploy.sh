#!/bin/bash
# One-time AWS bootstrap (run from your laptop with admin-ish credentials). Adjust the variables first.
set -euo pipefail
REGION=${REGION:-us-east-1}
API_URL=${API_URL:?set API_URL to the API endpoint reachable from Lambda}

for repo in api ui mlflow; do
  aws ecr describe-repositories --repository-names ai-governance-$repo --region $REGION >/dev/null 2>&1 \
    || aws ecr create-repository --repository-name ai-governance-$repo --region $REGION \
         --image-scanning-configuration scanOnPush=true
done

SNS_ARN=$(aws sns create-topic --name ai-governance-alerts --region $REGION --query TopicArn --output text)
echo "Subscribe your email:  aws sns subscribe --topic-arn $SNS_ARN --protocol email --notification-endpoint you@example.com"

# Lambda package
cd "$(dirname "$0")" && rm -f /tmp/lambda.zip && zip -j /tmp/lambda.zip lambda_handler.py
ROLE_ARN=${LAMBDA_ROLE_ARN:?create a Lambda execution role with cloudwatch:PutMetricData, sns:Publish and AWSLambdaBasicExecutionRole}
aws lambda create-function --function-name ai-governance-checks --runtime python3.12 --handler lambda_handler.handler \
  --zip-file fileb:///tmp/lambda.zip --role "$ROLE_ARN" --timeout 120 --region $REGION \
  --environment "Variables={API_URL=$API_URL,SNS_TOPIC_ARN=$SNS_ARN}" \
  || aws lambda update-function-code --function-name ai-governance-checks --zip-file fileb:///tmp/lambda.zip --region $REGION

# Hourly governance re-evaluation
aws events put-rule --name ai-governance-hourly --schedule-expression "rate(1 hour)" --region $REGION
FN_ARN=$(aws lambda get-function --function-name ai-governance-checks --query Configuration.FunctionArn --output text --region $REGION)
aws lambda add-permission --function-name ai-governance-checks --statement-id events-hourly --action lambda:InvokeFunction \
  --principal events.amazonaws.com --source-arn "$(aws events describe-rule --name ai-governance-hourly --query Arn --output text --region $REGION)" --region $REGION || true
aws events put-targets --rule ai-governance-hourly --targets "Id=1,Arn=$FN_ARN,Input={\"task\":\"governance_check\"}" --region $REGION

# CloudWatch alarms
aws cloudwatch put-metric-alarm --alarm-name ai-governance-high-risk --namespace AIGovernance --metric-name LambdaRiskScore \
  --statistic Maximum --period 3600 --evaluation-periods 1 --threshold 60 --comparison-operator GreaterThanThreshold \
  --alarm-actions "$SNS_ARN" --treat-missing-data notBreaching --region $REGION
aws cloudwatch put-metric-alarm --alarm-name ai-governance-api-p95-latency --namespace AIGovernance --metric-name api_latency_ms \
  --extended-statistic p95 --period 300 --evaluation-periods 2 --threshold 2000 --comparison-operator GreaterThanThreshold \
  --alarm-actions "$SNS_ARN" --treat-missing-data notBreaching --region $REGION
aws cloudwatch put-metric-alarm --alarm-name ai-governance-api-errors --namespace AIGovernance --metric-name api_errors \
  --statistic Sum --period 300 --evaluation-periods 1 --threshold 5 --comparison-operator GreaterThanThreshold \
  --alarm-actions "$SNS_ARN" --treat-missing-data notBreaching --region $REGION
echo "Done."
