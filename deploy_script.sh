#!/bin/bash

# Log in to Azure
az login --service-principal -u $AZURE_CLIENT_ID -p $AZURE_CLIENT_SECRET --tenant $AZURE_TENANT_ID

# Configure the Web App to use the Docker image
az webapp config container set --name $AZURE_APP_NAME --resource-group $AZURE_RESOURCE_GROUP --docker-custom-image-name $DOCKER_IMAGE
