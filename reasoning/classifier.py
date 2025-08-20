import argparse
import re
import numpy as np
import os
import json
import pandas as pd # Added for data handling and inspection

# Ensure these imports are correct based on your project structure
# You would need to provide or mock these if running this script completely standalone
# For this script to run as-is, `load_data` and `get_dataset_generator`
# from `utils` and `data` modules respectively are assumed to be available.
from utils import load_data
from data import get_dataset_generator

# Scikit-learn imports for classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# NLTK imports (ensure 'punkt' and 'stopwords' are downloaded)
from nltk.tokenize import word_tokenize
from nltk import ngrams, corpus

def get_args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-0.6B')
    parser.add_argument('--data_name', type=str, default='gpqa',
                        help='Choose among: gpqa, gsm8k, mmlu, icraft, imedqa')
    return parser

def preprocess_text(text):
    """Remove punctuation, extra whitespace, and convert to lowercase."""
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def create_stopwords():
    """Create a comprehensive set of stopwords."""
    # Ensure to import nltk and download stopwords if not already done
    stopwords = set()
    
    # Single character words
    stopwords.update({chr(i) for i in range(97, 123)})
    # Two character words
    stopwords.update({chr(i) + chr(j) for i in range(97, 123) for j in range(97, 123)})
    # Numbers
    stopwords.update({str(i) for i in range(10)})
    return stopwords

def classify_thinking_content(data):
    """
    Classifies thinking content as with or without context using n-grams and Random Forest.
    
    Args:
        data (list): A list of dictionaries, each containing 'thinking_content'
                     and 'thinking_content_without_context'.
    
    Returns:
        dict: A dictionary containing classification accuracy, report, confusion matrix,
              and top feature importances.
    """
    all_texts = []
    labels = []  # 1 for with_context, 0 for without_context

    # Collect thinking content and assign labels
    for row in data:
        # Only add if the content is not empty after preprocessing
        ctx_content = preprocess_text(row.get("thinking_content", ""))
        if ctx_content:
            all_texts.append(ctx_content)
            labels.append(1)  # With context

        no_ctx_content = preprocess_text(row.get("thinking_content_without_context", ""))
        if no_ctx_content:
            all_texts.append(no_ctx_content)
            labels.append(0)  # Without context

    if not all_texts or len(set(labels)) < 2:
        return {
            "classification_accuracy": "N/A (Insufficient data or only one class present)",
            "classification_report": "N/A",
            "top_feature_importances": []
        }

    # Vectorize the text data using TF-IDF
    # ngram_range=(2, 3) means we'll consider bigrams and trigrams
    # stop_words=list(create_stopwords()) removes common words
    # min_df=5 ignores terms that appear in fewer than 5 documents (helps with sparsity)
    vectorizer = TfidfVectorizer(ngram_range=(2, 4), stop_words=list(create_stopwords()), min_df=3)
    X = vectorizer.fit_transform(all_texts)
    y = np.array(labels)

    # Split data into training and testing sets
    # stratify=y ensures that the proportion of classes is the same in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Initialize and train the Random Forest Classifier
    # n_estimators=100: number of trees in the forest
    # random_state=42: for reproducibility
    # class_weight='balanced': automatically adjusts weights inversely proportional to class frequencies
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_classifier.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = rf_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    # Generate a detailed classification report
    report = classification_report(y_test, y_pred, target_names=['Without Context', 'With Context'], output_dict=True)

    # Get feature importances
    feature_names = vectorizer.get_feature_names_out()
    importances = rf_classifier.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    return {
        "classification_accuracy": accuracy,
        "classification_report": report,
        "top_feature_importances": feature_importance_df.head(20)['feature'].tolist()  # Top 20 most important n-grams
    }


def main():
    """Main function to run the classification."""
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Ensure NLTK data is available
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt')
    try:
        import nltk
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK 'stopwords' corpus...")
        nltk.download('stopwords')

    # Load data using your existing utility function
    # Note: For this script to run as a standalone, `load_data` must be defined
    # or mocked to return data in the expected format.
    data = load_data(args.model_name, args.data_name)

    if not data:
        print(f"No data to classify for model: {args.model_name}, dataset: {args.data_name}. Exiting.")
        return {"error": "No data found or loaded"}

    print("\n--- Running Classification Analysis ---")
    classification_results = classify_thinking_content(data)
    print("--- Classification Analysis Complete ---")

    # save results to a file
    output_dir = f"results/{args.model_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.data_name}_classification_results.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(classification_results, f, indent=4)
        
    return classification_results


if __name__ == "__main__":
    results = main()
    print("\n--- Classification Results Summary ---")
    print(json.dumps(results, indent=2))
