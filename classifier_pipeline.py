"""Pipeline to generate true labels."""

import pandas as pd 
import logging
from collections import Counter
from src.processing import Processor
from src.models import Classifier

classifier = Classifier()
processor = Processor()

logging.basicConfig(level=logging.INFO)

def generate_labels(df):
    """Generate labels for train data
    """
    label_lookup = {}
    for idx in range(len(df)):
        logging.info(f"Genarating label for index: {idx}")
        query, title, description, _ = processor.format_context(df, idx)
        current_output = classifier.chat_completion('task1', query, title, description)
        label_lookup[idx] = {
            'query': query,
            'title': title, 
            'description': description,
            'label': current_output.label,
            'explanation': current_output.explanation}
    return pd.DataFrame(label_lookup).T

def main():
    df_examples = pd.read_csv('data/sample_dataset.csv')
    # classify 3 query cases
    df1 = df_examples.query('query == "aa batteries 100 pack"').reset_index(drop=True)
    df2 = df_examples.query('query == "kodak photo paper 8.5 x 11 glossy"').reset_index(drop=True)
    df3 = df_examples.query('query == "dewalt 8v max cordless screwdriver kit, gyroscopic"').reset_index(drop=True)
    label_df1 = generate_labels(df1)
    logging.info(f"Labels for df1 generated: {label_df1.head()}")
    logging.info(f"Accuracy: {Counter(label_df1.label)['E']/len(label_df1)}")

    label_df2 = generate_labels(df2)
    logging.info(f"Labels for df2 generated: {label_df2.head()}")
    logging.info(f"Accuracy: {Counter(label_df2.label)['E']/len(label_df2)}")

    label_df3 = generate_labels(df3)
    logging.info(f"Labels for df3 generated: {label_df3.head()}")
    logging.info(f"Accuracy: {Counter(label_df3.label)['E']/len(label_df3)}")

    logging.info(f"Exporting labels...")
    label_df1.to_csv('data/label_df1.csv', index=False)
    label_df2.to_csv('data/label_df2.csv', index=False)
    label_df3.to_csv('data/label_df3.csv', index=False)
    logging.info(f"Export finished.")

if __name__ == "__main__":
    main()


