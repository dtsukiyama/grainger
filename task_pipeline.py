"""Pipeline to determine query correctness and generate reformulated query."""

import pandas as pd 
import logging
from collections import Counter
from src.processing import Processor
from src.models import Controller

controller = Controller()
processor = Processor()

logging.basicConfig(level=logging.INFO)

def evaluate_labels(df):
    """Generate labels for train data
    """
    label_lookup = {}
    for idx in range(len(df)):
        logging.info(f"Evaluate label for index: {idx}")
        query, title, description, label = processor.format_context(df, idx)
        task1_answer, task1_explanation, task2_exact_match, task2_explanation, task3_new_query = controller.run_tasks(query, title, description, label)
        label_lookup[idx] = {
            'query_id': df.loc[idx].query_id,
            'product_id': df.loc[idx].product_id,
            'query': df.loc[idx].query,
            'title': df.loc[idx].product_title,
            'query_is_correct': task1_answer,
            'explanation': task1_explanation, 
            'validation': task2_exact_match,
            'validation_explanation': task2_explanation,
            'reformulated_query': task3_new_query}
    return pd.DataFrame(label_lookup).T

def main():
    df_examples = pd.read_csv('data/sample_dataset.csv')
    # classify 3 query cases
    df1 = df_examples.query('query == "aa batteries 100 pack"').reset_index(drop=True)
    df2 = df_examples.query('query == "kodak photo paper 8.5 x 11 glossy"').reset_index(drop=True)
    df3 = df_examples.query('query == "dewalt 8v max cordless screwdriver kit, gyroscopic"').reset_index(drop=True)
    label_df1 = evaluate_labels(df1)
    logging.info(f"Evaluated Labels for df1: {label_df1.head()}")

    label_df2 = evaluate_labels(df2)
    logging.info(f"Evaluated Labels for df2: {label_df2.head()}")

    label_df3 = evaluate_labels(df3)
    logging.info(f"Evaluated Labels for df3: {label_df3.head()}")

    logging.info(f"Exporting Evaluated labels...")
    label_df1.to_csv('data/evaluated_labels_df1.csv', index=False)
    label_df2.to_csv('data/evaluated_labels_df2.csv', index=False)
    label_df3.to_csv('data/evaluated_labels_df3.csv', index=False)
    logging.info(f"Export finished.")

if __name__ == "__main__":
    main()


