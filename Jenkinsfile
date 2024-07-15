pipeline {
    agent any

    stages {
        stage('Clone repository') {
            steps {
                git 'https://github.com/Rudrank1/DiamondPriceModel.git'
            }
        }

        stage('Install dependencies') {
            steps {
                script {
                    sh 'pip install -r requirements.txt'
                }
            }
        }

        stage('Run script') {
            steps {
                script {
                    try {
                        sh 'app.py'
                    } catch (Exception e) {
                        error "Script execution failed: ${e.getMessage()}"
                    }
                }
            }
        }
    }
}
