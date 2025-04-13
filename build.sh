#!/bin/bash

# Stop on any error
set -e

# Define image name and tag
IMAGE_NAME="pipeline-runner"
IMAGE_TAG="latest"

# Display banner
echo "========================================"
echo "  Building Docker Image: $IMAGE_NAME:$IMAGE_TAG"
echo "========================================"

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo "Error: Dockerfile not found in current directory!"
    exit 1
fi

# Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME:$IMAGE_TAG .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Docker image built successfully: $IMAGE_NAME:$IMAGE_TAG"
    
    # Display usage instructions
    echo ""
    echo "Run pipelines with:"
    echo "./run_pipeline.sh classifier_pipeline"
    echo "./run_pipeline.sh task_pipeline"
    echo "./run_pipeline.sh eval_pipeline"
else
    echo "Error: Docker build failed."
    exit 1
fi