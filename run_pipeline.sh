#!/bin/bash

# Check if pipeline parameter was passed
if [ -z "$1" ]; then
  echo "Error: Please specify a pipeline to run (classifier_pipeline, task_pipeline, test_eval_pipeline, reformulated_query_pipeline)"
  echo "Usage: $0 <pipeline_name> [env_file]"
  exit 1
fi

# Validate the pipeline name
PIPELINE_NAME=$1
if [[ ! "$PIPELINE_NAME" =~ ^(classifier|task|test_eval|reformulated_query|test_reformulated_eval)_pipeline$ ]]; then
  echo "Error: Invalid pipeline name. Choose from: classifier_pipeline, task_pipeline, test_eval_pipeline, reformulated_query_pipeline, or test_reformuated_eval_pipeline"
  exit 1
fi

# Check for custom env file as second parameter, default to .env
ENV_FILE=${2:-.env}

# Check if env file exists
if [ ! -f "$ENV_FILE" ]; then
  echo "Warning: ENV file $ENV_FILE not found. Continue without env file? (y/n)"
  read -r response
  if [[ "$response" =~ ^[Nn]$ ]]; then
    exit 1
  fi
  # Run Docker without env file
  docker run -v "$(pwd):/app" pipeline-runner "$PIPELINE_NAME"
else
  # Run Docker with env file
  echo "Running $PIPELINE_NAME with environment from $ENV_FILE"
  docker run -v "$(pwd):/app" --env-file "$ENV_FILE" pipeline-runner "$PIPELINE_NAME"
fi