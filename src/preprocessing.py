import pandas as pd
import re
import argparse
import sys

# --- Load stopwords ---
def load_stopwords(filepath):
    try:
        with open(filepath, 'r') as f:
            return set(line.strip() for line in f)
    except FileNotFoundError:
        print(f"Stopwords file not found: {filepath}")
        return set()

# --- Tokenize and filter ---
def tokenize_and_filter(text, stopwords):
    if pd.isna(text):
        return []

    delimiter_chars = r'()\[\]{}.!?,;:+=\-_"\'`~#@&*%€$§<>^\\/'
    split_pattern = rf'[\s\d{re.escape(delimiter_chars)}]+'

    tokens = re.split(split_pattern, str(text).lower())
    return [
        token for token in tokens
        if token and token not in stopwords and len(token) > 1
    ]

# --- Main preprocessing ---
def preprocess(df, stopwords, text_columns):
    df_copy = df.copy()
    for col in text_columns:
        if col in df_copy.columns:
            df_copy[col + '_tokens'] = df_copy[col].apply(lambda x: tokenize_and_filter(x, stopwords))
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")
    return df_copy

# --- Script entry point ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to the input CSV file')
    parser.add_argument('--output', default='chi_input.csv', help='Path to save the processed CSV')
    parser.add_argument('--stopwords', default='../Assignment_1_Assets/stopwords.txt', help='Path to stopwords file')
    args = parser.parse_args()

    print(f"Loading data from {args.input}")
    try:
        df = pd.read_json(args.input, lines = True)
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        sys.exit(1)

    stopwords = load_stopwords(args.stopwords)

    print("Preprocessing...")
    df_processed = preprocess(df, stopwords, text_columns=['reviewText', 'summary'])

    try:
        df_processed.to_csv(args.output, index=False, columns=['reviewText_tokens', 'category'])
        print(f"Processed data saved to {args.output}")
    except Exception as e:
        print(f"Error saving output CSV: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
