import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

# =========================
# Download NLTK resources safely
# =========================
def download_nltk_resources():
    resources = ['punkt', 'punkt_tab', 'stopwords']
    for resource in resources:
        try:
            if resource in ['punkt', 'punkt_tab']:
                nltk.data.find(f'tokenizers/{resource}')
            else:
                nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)

download_nltk_resources()

# =========================
# Global objects (performance optimized)
# =========================
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# =========================
# Logging setup
# =========================
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(os.path.join(log_dir, 'data_preprocessing.log'))
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# =========================
# Text Transformation
# =========================
def transform_text(text):
    try:
        if pd.isnull(text):
            return ""

        text = text.lower()
        tokens = nltk.word_tokenize(text)

        tokens = [word for word in tokens if word.isalnum()]
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [ps.stem(word) for word in tokens]

        return " ".join(tokens)

    except Exception as e:
        logger.error("Error in transform_text: %s", e)
        return ""

# =========================
# Data Preprocessing
# =========================
def preprocess_df(df, text_column='text', target_column='target'):
    try:
        logger.debug('Starting preprocessing for DataFrame')

        # Encode target
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        # Remove duplicates
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')

        # Transform text
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')

        return df

    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during preprocessing: %s', e)
        raise

# =========================
# Main Pipeline
# =========================
def main(text_column='text', target_column='target'):
    try:
        # Load data
        train_path = './data/raw/train.csv'
        test_path = './data/raw/test.csv'

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        logger.debug('Data loaded properly')

        # Preprocess
        train_processed = preprocess_df(train_data, text_column, target_column)
        test_processed = preprocess_df(test_data, text_column, target_column)

        # Save processed data
        output_path = os.path.join("./data", "interim")
        os.makedirs(output_path, exist_ok=True)

        train_processed.to_csv(os.path.join(output_path, "train_processed.csv"), index=False)
        test_processed.to_csv(os.path.join(output_path, "test_processed.csv"), index=False)

        logger.debug('Processed data saved to %s', output_path)

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete data transformation: %s', e)
        print(f"Error: {e}")

# =========================
# Entry Point
# =========================
if __name__ == '__main__':
    main()