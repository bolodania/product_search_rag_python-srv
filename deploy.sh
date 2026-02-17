
#!/bin/bash

# ===============================
# Deployment Script for CF App
# ===============================
# - HANA config: uses UPVs (user-provided service)
# - AI Core: uses VCAP (managed service)
#
# This script ensures the required xsuaa service exists for authentication.
# HANA credentials must be provided as a user-provided service (UPV) and bound in the manifest.yml.
# No local config files are needed for deployment.

XSUAA_SERVICE_NAME="product_search_rag-python-srv-uaa"
XSUAA_SERVICE_KEY_NAME="uaa-service-key"

# Ensure xsuaa service exists
cf create-service xsuaa application "$XSUAA_SERVICE_NAME" || echo "xsuaa service creation triggered..."

# Wait for xsuaa service to be ready
get_service_status() {
  cf service "$XSUAA_SERVICE_NAME" | awk '/status:/{print $2, $3; exit}'
}

wait_for_service() {
  while true; do
    STATUS=$(get_service_status)
    if [[ "$STATUS" == "create succeeded" || "$STATUS" == "update succeeded" ]]; then
      return 0
    fi
    echo "xsuaa service is still in progress (current status: $STATUS)... checking again in 10 seconds"
    sleep 10
  done
}

CURRENT_STATUS=$(get_service_status)
if [[ "$CURRENT_STATUS" != "create succeeded" && "$CURRENT_STATUS" != "update succeeded" ]]; then
  wait_for_service
fi

# Create xsuaa service key if it doesn't exist
cf create-service-key "$XSUAA_SERVICE_NAME" "$XSUAA_SERVICE_KEY_NAME" || echo "xsuaa service key creation triggered..."

echo "Assuming HANA UPV and AI Core VCAP are already created and bound in manifest.yml."
echo "If not, create HANA UPV with:"
echo "  cf create-user-provided-service <hana-upv-name> -p '{\"HANA_HOST\":\"<host>\",\"HANA_PORT\":\"<port>\",\"HANA_USER\":\"<user>\",\"HANA_PASSWORD\":\"<password>\",\"HANA_SCHEMA\":\"<schema>\"}'"
echo "Then bind it in manifest.yml under 'services'."

# Deploy the application
cf push
