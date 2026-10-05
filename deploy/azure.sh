#!/usr/bin/env bash
# Hosts the model server on Azure Container Apps. Idempotent: rerun it to
# re-upload the checkpoint or pick up a new image.
#
#   bash deploy/azure.sh
#
# Needs `az login` first, and the image built by
# .github/workflows/model-image.yml (public on ghcr.io — it holds code only).
#
# What it makes, all in one resource group so `az group delete -n $RG` removes
# every trace:
#   - a storage account with a private file share holding best.pth and the
#     best.measured-*.json reports, mounted read-only at /mnt/checkpoints;
#   - a Container Apps environment (no Log Analytics workspace: it bills per GB);
#   - the app: 2 vCPU / 4 GiB, scales to zero when idle, so it costs nothing
#     between uploads and the first request after a quiet spell waits for it to
#     start (pull the image, load 1.26 GB of weights).
#
# /predict requires a bearer token (MODEL_API_TOKEN in app.py). The script
# generates it on first run and prints it; set the same value as
# MODEL_API_TOKEN on Vercel, with MODEL_API_URL as the printed URL.
set -euo pipefail

AZ=${AZ:-$HOME/tools/azcli/bin/az}
RG=${RG:-soundsentinal}
LOC=${LOC:-australiaeast}
ENV_NAME=${ENV_NAME:-soundsentinal-env}
APP=${APP:-soundsentinal-model}
SHARE=checkpoints
IMAGE=${IMAGE:-ghcr.io/mohidkz05/soundsentinal-model:latest}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CKPT="$ROOT/ai_model/checkpoints"

"$AZ" config set extension.use_dynamic_install=yes_without_prompt --only-show-errors
for ns in Microsoft.App Microsoft.Storage; do
  "$AZ" provider register -n "$ns" --wait --only-show-errors
done

"$AZ" group create -n "$RG" -l "$LOC" -o none

# Storage account names are global and 3-24 lowercase alphanumerics; derive a
# stable one from the subscription so reruns find the same account.
SUB=$("$AZ" account show --query id -o tsv)
SA="sentinal$(echo -n "$SUB" | sha1sum | cut -c1-12)"
"$AZ" storage account create -n "$SA" -g "$RG" -l "$LOC" --sku Standard_LRS \
  --kind StorageV2 --min-tls-version TLS1_2 --allow-blob-public-access false -o none
KEY=$("$AZ" storage account keys list -n "$SA" -g "$RG" --query '[0].value' -o tsv)
"$AZ" storage share-rm create --storage-account "$SA" -g "$RG" -n "$SHARE" --quota 5 -o none

# Only what app.py reads: the served checkpoint and its measured error rates.
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
ln "$CKPT/best.pth" "$STAGE/best.pth" 2>/dev/null || cp "$CKPT/best.pth" "$STAGE/"
cp "$CKPT"/best.measured-*.json "$STAGE/"
"$AZ" storage file upload-batch --account-name "$SA" --account-key "$KEY" \
  -d "$SHARE" -s "$STAGE" --max-connections 8 -o none

"$AZ" containerapp env show -n "$ENV_NAME" -g "$RG" -o none 2>/dev/null \
  || "$AZ" containerapp env create -n "$ENV_NAME" -g "$RG" -l "$LOC" \
       --logs-destination none -o none
"$AZ" containerapp env storage set -n "$ENV_NAME" -g "$RG" --storage-name ckpt \
  --azure-file-account-name "$SA" --azure-file-account-key "$KEY" \
  --azure-file-share-name "$SHARE" --access-mode ReadOnly -o none

# SKIP_APP=1 stops here: storage and environment only, e.g. before the image
# has been built for the first time.
if [ -n "${SKIP_APP:-}" ]; then echo "Infrastructure ready; app skipped."; exit 0; fi

# Keep the token across reruns; generate it once.
TOKEN=$("$AZ" containerapp secret show -n "$APP" -g "$RG" --secret-name model-api-token \
          --query value -o tsv 2>/dev/null || true)
TOKEN=${TOKEN:-$(openssl rand -hex 32)}
ENV_ID=$("$AZ" containerapp env show -n "$ENV_NAME" -g "$RG" --query id -o tsv)

cat > "$STAGE/app.yaml" <<YAML
location: $LOC
properties:
  managedEnvironmentId: $ENV_ID
  configuration:
    secrets:
      - name: model-api-token
        value: $TOKEN
    ingress:
      external: true
      targetPort: 8000
      transport: http
  template:
    containers:
      - name: model
        image: $IMAGE
        resources: { cpu: 2.0, memory: 4Gi }
        env:
          - name: MODEL_API_TOKEN
            secretRef: model-api-token
        volumeMounts:
          - volumeName: ckpt
            mountPath: /mnt/checkpoints
        probes:
          # Loading the weights takes a while; don't route traffic or restart
          # the container until /health answers.
          - type: Startup
            httpGet: { path: /health, port: 8000 }
            periodSeconds: 5
            failureThreshold: 48
    volumes:
      - name: ckpt
        storageType: AzureFile
        storageName: ckpt
    scale:
      minReplicas: 0
      maxReplicas: 1
YAML

if "$AZ" containerapp show -n "$APP" -g "$RG" -o none 2>/dev/null; then
  "$AZ" containerapp update -n "$APP" -g "$RG" --yaml "$STAGE/app.yaml" -o none
else
  "$AZ" containerapp create -n "$APP" -g "$RG" --yaml "$STAGE/app.yaml" -o none
fi

FQDN=$("$AZ" containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)
echo
echo "MODEL_API_URL=https://$FQDN"
echo "MODEL_API_TOKEN=$TOKEN"
