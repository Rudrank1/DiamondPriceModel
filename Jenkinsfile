pipeline {
    agent any

    environment {
        AZURE_WEBAPP_NAME = 'Diamond-Price-Prediction-Model'
        AZURE_RESOURCE_GROUP = 'appgroup'
        AZURE_CREDENTIALS_ID = '222'
        LOCATION = 'Australia Central'
        APP_SERVICE_PLAN = 'ASP-appgroup-a2df (F1: 1)'
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

        stage('Install Azure CLI') {
            steps {
                script {
                    sh '''
                    curl -sL https://aka.ms/InstallAzureCLIDeb | bash
                    '''
                }
            }
        }

        stage('Deploy to Azure') {
            steps {
                withCredentials([azureServicePrincipal(credentialsId: "${env.AZURE_CREDENTIALS_ID}")]) {
                    script {
                        sh '''
                        az webapp up --name ${env.AZURE_WEBAPP_NAME} \
                                     --resource-group ${env.AZURE_RESOURCE_GROUP} \
                                     --sku F1 \
                                     --location ${env.LOCATION} \
                                     --plan ${env.APP_SERVICE_PLAN} \
                                     --runtime "PYTHON|3.9"
                        '''
                    }
                }
            }
        }
    }
}
