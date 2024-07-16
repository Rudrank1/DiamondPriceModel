pipeline {
    agent any

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

        stage('Run script') {
            steps {
                script {
                    try {
                        sh '/Users/rudrank/Library/Python/3.9/bin/streamlit run Diamond/app.py'
                    } catch (Exception e) {
                        error "Script execution failed: ${e.getMessage()}"
                    }
                }
            }
        }
    }
}
