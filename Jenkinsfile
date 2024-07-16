pipeline {
    agent any

    stages {
        stage('Check Streamlit Location') {
            steps {
                script {
                    def streamlitInfo = sh(script: 'pip3 show streamlit | grep Location', returnStdout: true).trim()
                    echo "Streamlit is located at: ${streamlitInfo}"
                }
            }
        }
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

        stage('Run script') {
            steps {
                script {
                    try {
                        sh 'streamlit run Diamond/app.py'
                    } catch (Exception e) {
                        error "Script execution failed: ${e.getMessage()}"
                    }
                }
            }
        }
    }
}
