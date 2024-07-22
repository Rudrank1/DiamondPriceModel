pipeline {
    agent any

    environment {
        AZURE_WEBAPP_NAME = 'Diamond-Price-Prediction-Model'
        AZURE_RESOURCE_GROUP = 'appgroup'
        AZURE_CREDENTIALS_ID = '222'
        AZURE_SUBSCRIPTION_ID = 'c8661fb5-0aff-452a-8693-e440206a5a0b'
    }

    stages {
        stage('Clone repository') {
            steps {
                git branch: 'main', url: 'https://github.com/Rudrank1/DiamondPriceModel.git'
            }
        }

        stage('Install dependencies') {
            steps {
                script {
                    sh 'pip3 install -r Diamond/requirements.txt'
                }
            }
        }

        stage('Deploy to Azure') {
            steps {
                script {
                    withCredentials([azureServicePrincipal(credentialsId: "${env.AZURE_CREDENTIALS_ID}")]) {
                        sh """
                        az webapp up --name ${env.AZURE_WEBAPP_NAME} \
                                     --resource-group ${env.AZURE_RESOURCE_GROUP} \
                                     --sku F1 \
                                     --location centralus \
                                     --plan myAppServicePlan \
                                     --runtime "PYTHON|3.9" \
                                     --subscription ${env.AZURE_SUBSCRIPTION_ID}
                        """
                    }
                }
            }
        }
    }
}
