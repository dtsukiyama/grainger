#!/bin/bash

# Check if pipeline parameter was passed
if [ -z "$1" ]; then
  echo "Error: Please specify a pipeline to run (classifier_pipeline, task_pipeline, test_eval_pipeline, reformulated_query_pipeline or test_reformuated_eval)"
  exit 1
fi

# Validate the pipeline name
PIPELINE_NAME=$1
if [[ ! "$PIPELINE_NAME" =~ ^(classifier|task|test_eval|reformulated_query|test_reformulated_eval)_pipeline$ ]]; then
  echo "Error: Invalid pipeline name. Choose from: classifier_pipeline, task_pipeline, eval_pipeline, reformulated_query_pipeline, or test_reformuated_eval_pipeline"
  exit 1
fi

# Run the specified pipeline
if [[ "$PIPELINE_NAME" == "test_eval_pipeline" || "$PIPELINE_NAME" == "test_reformulated_eval_pipeline" ]]; then
  echo "Running test evaluation pipeline with deepeval..."
  # Use the -n flag to specify number of concurrent tests (4 in this case)
  deepeval test run ${PIPELINE_NAME}.py -n 4
else
  # For other pipelines, run them as normal Python scripts
  echo "Running $PIPELINE_NAME.py..."
  python "$PIPELINE_NAME.py"
fi