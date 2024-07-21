pipeline {
    agent {
        label 'docker'
    }

    environment {
        DOCKER_HUB_CREDENTIALS = credentials('111')
        DOCKER_IMAGE = 'rudy12/diamond-prediction-model:latest'
        AZURE_CREDENTIALS = credentials('222')
        AZURE_WEB_APP_NAME = 'diamond-price-prediction-model'
        AZURE_RESOURCE_GROUP = 'testgroup'
    }

    stages {
        stage('Clone repository') {
            steps {
                git branch: 'main', url: 'https://github.com/Rudrank1/DiamondPriceModel.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    sh 'docker build -t $DOCKER_IMAGE .'
                }
            }
        }

        stage('Push Docker Image to Docker Hub') {
            steps {
                script {
                    sh """
                        echo $DOCKER_HUB_CREDENTIALS_PSW | docker login -u $DOCKER_HUB_CREDENTIALS_USR --password-stdin
                        docker push $DOCKER_IMAGE
                    """
                }
            }
        }

        stage('Deploy to Azure Web App') {
            steps {
                script {
                    withCredentials([azureServicePrincipal(credentialsId: '222')]) {
                        sh """
                            az login --service-principal -u $AZURE_CREDENTIALS_USR -p $AZURE_CREDENTIALS_PSW --tenant $AZURE_CREDENTIALS_TENANT
                            az webapp config container set --name $AZURE_WEB_APP_NAME --resource-group $AZURE_RESOURCE_GROUP --docker-custom-image-name $DOCKER_IMAGE
                        """
                    }
                }
            }
        }
    }
}
