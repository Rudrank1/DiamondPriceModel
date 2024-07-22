pipeline {
    agent any

    environment {
        AZURE_WEBAPP_NAME = 'Diamond-Price-Prediction-Model'
        AZURE_RESOURCE_GROUP = 'appgroup'
        AZURE_CREDENTIALS_ID = '222'
        LOCATION = 'australiacentral'
        APP_SERVICE_PLAN = 'ASP-appgroup-a2df (F1: 1)'
        AZ_PATH = '/usr/local/bin/az'
    }

    stages {
        stage('Clone repository') {
            steps {
                git branch: 'main', url: 'https://github.com/Rudrank1/DiamondPriceModel.git'
            }
        }

        stage('Install dependencies') {
            steps {
                sh 'pip3 install -r Diamond/requirements.txt'
            }
        }

        stage('Deploy to Azure') {
            steps {
                withCredentials([azureServicePrincipal(credentialsId: "${env.AZURE_CREDENTIALS_ID}")]) {
                    script {
                        // Authenticate with Azure CLI using the Service Principal
                        sh '''
                        ${AZ_PATH} login --service-principal -u $AZURE_CLIENT_ID -p $AZURE_CLIENT_SECRET --tenant $AZURE_TENANT_ID
                        '''

                        // Deploy the web app
                        sh '''
                        ${AZ_PATH} webapp up --name $AZURE_WEBAPP_NAME \
                                             --resource-group $AZURE_RESOURCE_GROUP \
                                             --sku F1 \
                                             --location $LOCATION \
                                             --plan "$APP_SERVICE_PLAN" \
                                             --runtime "PYTHON|3.9"
                        '''
                    }
                }
            }
        }
    }
}
