#!/bin/bash

SERVICE_NAME="product_search_rag_YOUR_NUMBER-uaa"
SERVICE_KEY_NAME="uaa-service-key"  # Name of the service key

# Create the service if it doesn't exist
cf create-service xsuaa application "$SERVICE_NAME" || echo "Service creation triggered..."

# Function to check service status
get_service_status() {
  cf service "$SERVICE_NAME" | awk '/status:/{print $2, $3; exit}'
}

# Function to wait until the service is ready
wait_for_service() {
  while true; do
    STATUS=$(get_service_status)
    
    if [[ "$STATUS" == "create succeeded" || "$STATUS" == "update succeeded" ]]; then
      return 0
    fi

    echo "Service is still in progress (current status: $STATUS)... checking again in 10 seconds"
    sleep 10
  done
}

# Wait for the service to be ready
CURRENT_STATUS=$(get_service_status)

if [[ "$CURRENT_STATUS" != "create succeeded" && "$CURRENT_STATUS" != "update succeeded" ]]; then
  wait_for_service
fi

# Create the service key if it doesn't exist
cf create-service-key "$SERVICE_NAME" "$SERVICE_KEY_NAME" || echo "Service key creation triggered..."

# Deploy the application
cf push
