pipeline {
    agent any

    environment {
        // Docker Hub image (change to username/repo if needed)
        REGISTRY        = "swm-ai-backend"
        CONTAINER_NAME  = "swm-ai-backend"
        HOST_PORT       = "5001"
        CONTAINER_PORT  = "5001"
    }

    triggers {
        githubPush()
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/Chanri2025/Garbage_Backend.git'
            }
        }

        stage('Prepare .env (from Jenkins secret file)') {
            steps {
                // Jenkins credentials: "Secret file" type
                withCredentials([file(credentialsId: 'swm-ai-backend-env-file', variable: 'ENV_FILE')]) {
                    sh '''
                      echo "Copying env file from Jenkins credential..."
                      cp "$ENV_FILE" .env
                    '''
                }
            }
        }

        stage('Build Docker image') {
            steps {
                script {
                    def imageTag = "${env.BUILD_NUMBER}"
                    sh """
                      docker build \
                        -t ${REGISTRY}:${imageTag} \
                        -t ${REGISTRY}:latest \
                        .
                    """
                }
            }
        }

        stage('Push Docker image') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-swm-ai-backend',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh '''
                      echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                      docker push ${REGISTRY}:${BUILD_NUMBER}
                      docker push ${REGISTRY}:latest
                      docker logout
                    '''
                }
            }
        }

        stage('Deploy (pull & restart container)') {
            steps {
                withCredentials([file(credentialsId: 'swm-ai-backend-env-file', variable: 'ENV_FILE')]) {
                    sh '''
                      echo "Pulling latest image..."
                      docker pull ${REGISTRY}:latest

                      echo "Stopping old container if exists..."
                      docker stop ${CONTAINER_NAME} || true

                      echo "Removing old container if exists..."
                      docker rm ${CONTAINER_NAME} || true

                      echo "Starting new container on port ${HOST_PORT}..."
                      docker run -d \
                        --name ${CONTAINER_NAME} \
                        --restart always \
                        -p ${HOST_PORT}:${CONTAINER_PORT} \
                        --env-file "$ENV_FILE" \
                        ${REGISTRY}:latest

                      echo "Deployment complete."
                    '''
                }
            }
        }
    }

    post {
        always {
            sh 'docker image prune -f || true'
        }
    }
}
