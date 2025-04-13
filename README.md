# Overview

## Task Review

In this task you will identify examples where the label “E” is misapplied for the following
queries: “aa batteries 100 pack”, “kodak photo paper 8.5 x 11 glossy”, and “dewalt 8v max
cordless screwdriver kit, gyroscopic”. When you find that the query is incorrect for that
product, reformulate what an exact query for that product would be.

Your final output should be a table with query_id, product_id, then whether or not your
approach found that match to be accurate, and finally where inaccurate a refolumation of
the query that would make the relationship label “E.”

## Approach

I created a subset of the using the following queries only:

1. “aa batteries 100 pack”
2. “kodak photo paper 8.5 x 11 glossy”
3. “dewalt 8v max cordless screwdriver kit, gyroscopic”

We will refer to this as the training data.

I created a prompt to correctly classify queries according to the "E" label criteria; then I verifies these labels manually.

These labels can be created by running (follow Docker setup instructions):

`./run_pipeline.sh classifier_pipeline`

This generates label_df1.csv, label_df2.csv, label_df3.csv.

These datasets serve as the ground truth labels.

Next I created a multi-step approach to verify label correctness in the original label from the training data.
The next step is to validate the output of the preceding step, finally the third step is to generate a reformulated query if necessary. This generates the final output which contains the query_id, product_id, query_is_correct (whether or not the
approach found that match to be accurate), and the reformulated query. 

The final output can be generated by running:

`./run_pipeline.sh task_pipeline`

This generates evaluated_labels_df1.csv, evaluated_labels_df2.csv, evaluated_labels_df2.csv


Finally, I run an evaluation of my approach by using the ground truth labels as the expected output and the final output labels (query_is_correct) as the actual output labels. 

This can be run by doing:

`./run_pipeline.sh test_eval_pipeline`

My run indicates that all test cases pass. 

# Validating Reformulated Queries

To validate reformulated queries, we run the classifier again to label these new reformulated queries, we should expect that they get labeled "E". 

To generate these reformulated queries labels run:

`./run_pipeline.sh reformulated_query_pipeline`

To evaluate the performance of these reformulated queries, run:

`./run_pipeline.sh test_reformulated_eval_pipeline`

My test show that all reformulated queries have been labeled "E."


# Gotchas

These pipelines must be run sequentially because consequent pipelines rely on preceding pipeline output files.

# Caveats

I did not use local LLMs because if the additional overhead involved. 

I wanted to test the approach against unseen queries; however i determined I did not have enough time to do that.

# Docker Pipeline Runner

This setup allows you to run your three pipelines within Docker, using environment variables from a `.env` file.

# Setup Steps

Make your scripts executable

`chmod +x build.sh`

`chmod +x run_pipeline.sh`

`chmod +x docker-entrypoint.sh`

Build the Docker image

`./build.sh`

# Run your pipelines

Create a .env file with your OpenAI API key

Run with .env file

# Classifier pipeline

`./run_pipeline.sh classifier_pipeline`

# Final Output pipeline

`./run_pipeline.sh task_pipeline`

# Evaluation pipeline
`./run_pipeline.sh test_eval_pipeline`

# Reformulated Query Labels

`./run_pipeline.sh reformulated_query_pipeline`

# Reformulated Query Evaluation

`./run_pipeline.sh test_reformulated_eval_pipeline`


# I don't wnat to use Docker

Okay, if you don't want to use Docker then create a virtual environment:

```
conda create --name myenv python=3.12
pip install -r requirements.txt
```

Set your OpenAI API Key as an environment variable.

Run classifier pipeline:

`python classifier_pipeline.py`

Run final output pipeline:

`python task_pipeline.py`

Run final output pipeline evaluation:

`deepeval test run test_eval_pipeline.py -n 4`

Run Reformulated Query label pipeline:

`python reformulated_query_pipeline.py`

Run Reformulated Query Label Evaluation:

`deepeval test run test_reformulated_eval_pipeline.py -n 4`

# What Each File Does

Dockerfile: Defines how to build your Docker image

docker-entrypoint.sh: The script that runs INSIDE the container - it receives the pipeline name and runs the appropriate Python script

run_pipeline.sh: The script you run FROM YOUR HOST - it launches the Docker container with the right parameters

build_docker.sh: A convenient script to build your Docker image

Environment Variables

The .env file will be passed into the Docker container, making your environment variables available to your Python scripts. Example .env file:

OPENAI_API_KEY=your_api_key_here

## Directory Structure

```
repo/
├── data/
├── src/
├── classifier_pipeline.py
├── task_pipeline.py
├── test_eval_pipeline.py
├── reformulated_query_pipeline.py
├── test_reformulated_eval_pipeline.py
├── requirements.txt
├── Dockerfile
├── build.sh
├── docker-entrypoint.sh
├── run_pipeline.sh
└── .env
```
