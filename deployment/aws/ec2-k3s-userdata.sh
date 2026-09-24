#!/bin/bash
# EC2 user-data (Ubuntu 22.04/24.04, t3.large+). Installs k3s (single-node Kubernetes) and the CloudWatch agent.
# Attach an instance profile with deployment/aws/iam-ec2-policy.json.
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y curl unzip jq
curl -sfL https://get.k3s.io | sh -s - --write-kubeconfig-mode 644
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o /tmp/awscli.zip && unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install

# CloudWatch agent (memory/disk/CPU + container logs)
curl -fsSL "https://amazoncloudwatch-agent.s3.amazonaws.com/ubuntu/$( [ "$(uname -m)" = aarch64 ] && echo arm64 || echo amd64 )/latest/amazon-cloudwatch-agent.deb" -o /tmp/cw.deb
dpkg -i /tmp/cw.deb
# copy deployment/aws/cloudwatch-agent.json to the instance (or bake it into the AMI), then:
#   /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/opt/aws/cloudwatch-agent.json -s

# ECR image-pull secret, refreshed every 6h (ECR tokens last 12h)
cat > /usr/local/bin/ecr-refresh.sh <<'SCRIPT'
#!/bin/bash
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
REGION=${AWS_REGION:-us-east-1}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
kubectl -n ai-governance delete secret ecr-pull --ignore-not-found
kubectl -n ai-governance create secret docker-registry ecr-pull \
  --docker-server="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com" --docker-username=AWS \
  --docker-password="$(aws ecr get-login-password --region $REGION)"
SCRIPT
chmod +x /usr/local/bin/ecr-refresh.sh
echo "0 */6 * * * root /usr/local/bin/ecr-refresh.sh" > /etc/cron.d/ecr-refresh
