""""""

import pandas as pd 
import logging
from collections import Counter
from src.processing import Processor
from src.models import Classifier

classifier = Classifier()
processor = Processor()

logging.basicConfig(level=logging.INFO)

def generate_labels(df, reformulated):
    """Generate labels for train data
    """
    label_lookup = {}
    for idx in range(len(df)):
        if not reformulated.loc[idx].query_is_correct:
            logging.info(f"Generating label for index: {idx}")
            _, title, description, _ = processor.format_context(df, idx)
            reformulated_query = reformulated.loc[idx].reformulated_query
            current_output = classifier.chat_completion('task1', reformulated_query, title, description)
            label_lookup[idx] = {
                'query': reformulated_query,
                'title': title, 
                'description': description,
                'label': current_output.label,
                'explanation': current_output.explanation}
    return pd.DataFrame(label_lookup).T

def main():
    df_examples = pd.read_csv('data/sample_dataset.csv')
    # classify 3 query cases
    df1 = df_examples.query('query == "aa batteries 100 pack"').reset_index(drop=True)
    reformulated1 = pd.read_csv('data/evaluated_labels_df1.csv')

    df2 = df_examples.query('query == "kodak photo paper 8.5 x 11 glossy"').reset_index(drop=True)
    reformulated2 = pd.read_csv('data/evaluated_labels_df2.csv')

    df3 = df_examples.query('query == "dewalt 8v max cordless screwdriver kit, gyroscopic"').reset_index(drop=True)
    reformulated3 = pd.read_csv('data/evaluated_labels_df3.csv')

    label_df1 = generate_labels(df1, reformulated1)
    logging.info(f"Labels for df1 generated: {label_df1.head()}")

    label_df2 = generate_labels(df2, reformulated2)
    logging.info(f"Labels for df2 generated: {label_df2.head()}")

    label_df3 = generate_labels(df3, reformulated3)
    logging.info(f"Labels for df3 generated: {label_df3.head()}")

    logging.info(f"Exporting labels...")
    label_df1.to_csv('data/reformulated_label_df1.csv', index=False)
    label_df2.to_csv('data/reformulated_label_df2.csv', index=False)
    label_df3.to_csv('data/reformulated_label_df3.csv', index=False)
    logging.info(f"Export finished.")

if __name__ == "__main__":
    main()


