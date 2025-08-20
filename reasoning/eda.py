"""Analysis script for thinking content comparison."""

# import nltk
# nltk.download('stopwords')

import argparse
import re
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk import ngrams, corpus
from scipy.stats import ttest_rel
import numpy as np
import os
import json
from utils import read_jsonl



def load_data(model_name, data_name, results_path='results'):
    """Load data from the specified path."""
    path = f"{results_path}/{model_name}/{data_name}_results.jsonl"
    return read_jsonl(path)

# Ensure these imports are correct based on your project structure
from data import get_dataset_generator
from detect.abstention_keywords import ABSTENTION_KEYWORDS, ABSTENTION_KEYWORDS_WITH_ASSUMPTION
from detect.uncertainty_keywords import uncertainty_keywords_dict

def get_args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-0.6B')
    parser.add_argument('--data_name', type=str, default='gpqa',
                        help='Choose among: gpqa, gsm8k, mmlu, icraft, imedqa')
    parser.add_argument('--results_path', type=str, default='results',
                        help='Path to save results. Default is "results".')
    return parser


def analyze_answer_coverage(data, extract_answer):
    """Analyze what percentage of questions were answered."""
    answered_count = []
    answered_count_without_context = []
    
    for row in data:
        content = row.get("content", "")
        content_without_context = row.get("content_without_context", "")

        answered_count.append(extract_answer(content) is not None)
        answered_count_without_context.append(extract_answer(content_without_context) is not None)
    
    answered_percentage = sum(answered_count) / len(answered_count) * 100
    answered_percentage_without_context = sum(answered_count_without_context) / len(answered_count_without_context) * 100
    
    return answered_percentage, answered_percentage_without_context

def detect_abstention(model_answer, keywords=ABSTENTION_KEYWORDS_WITH_ASSUMPTION):
    """Search in response for keywords."""
    for keyword in keywords:
        if keyword.lower() in model_answer.lower():
            return True
    return False


def preprocess_text(text):
    """Remove punctuation, extra whitespace, and convert to lowercase."""
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def analyze_thinking_lengths(data):
    """Analyze and plot the distribution of thinking content lengths, including a t-test."""
    thinking_lengths = [len(row.get("thinking_content", "")) for row in data]
    thinking_lengths_without_context = [len(row.get("thinking_content_without_context", "")) for row in data]

    avg_thinking_length = np.mean(thinking_lengths) if thinking_lengths else 0
    avg_thinking_length_without_context = np.mean(thinking_lengths_without_context) if thinking_lengths_without_context else 0

    t_stat_length, p_value_length = None, None
    if len(thinking_lengths) > 1 and len(thinking_lengths) == len(thinking_lengths_without_context):
        t_stat_length, p_value_length = ttest_rel(thinking_lengths, thinking_lengths_without_context)

    return avg_thinking_length, avg_thinking_length_without_context, t_stat_length, p_value_length


def count_uncertainty(text):
    """Count the occurrences of uncertainty phrases in the text."""
    count = 0
    # Ensure text is lowercase for matching
    text_lower = text.lower()
    uncertainty_phrases = uncertainty_keywords_dict.get("doubt_speculation", [])
    for phrase in uncertainty_phrases:
        count += text_lower.count(phrase)
    return count


def analyze_uncertainty(data):
    """Analyze uncertainty phrases and perform statistical testing."""
    proportions_ctx = []
    proportions_no_ctx = []

    for row in data:
        thinking_content = row.get("thinking_content", "")
        words_ctx = thinking_content.split()
        if words_ctx:
            proportions_ctx.append(count_uncertainty(thinking_content) / len(words_ctx))
        else:
            proportions_ctx.append(0)

        thinking_content_no_ctx = row.get("thinking_content_without_context", "")
        words_no_ctx = thinking_content_no_ctx.split()
        if words_no_ctx:
            proportions_no_ctx.append(count_uncertainty(thinking_content_no_ctx) / len(words_no_ctx))
        else:
            proportions_no_ctx.append(0)

    t_stat, p_value = None, None
    if len(proportions_ctx) > 1 and len(proportions_ctx) == len(proportions_no_ctx):
        t_stat, p_value = ttest_rel(proportions_ctx, proportions_no_ctx)

    avg_proportion_ctx = np.mean(proportions_ctx) if proportions_ctx else 0
    avg_proportion_no_ctx = np.mean(proportions_no_ctx) if proportions_no_ctx else 0

    return avg_proportion_ctx, avg_proportion_no_ctx, t_stat, p_value


def analyze_abstention(data):
    """Analyze the percentage of abstention responses."""
    abstention_count = sum(1 for row in data if detect_abstention(row.get("content", "")))
    abstention_count_without_context = sum(1 for row in data if detect_abstention(row.get("content_without_context", "")))

    total = len(data)
    abstention_percentage = (abstention_count / total * 100) if total > 0 else 0
    abstention_percentage_without_context = (abstention_count_without_context / total * 100) if total > 0 else 0

    return abstention_percentage, abstention_percentage_without_context


def analyze_answer_accuracy(data, evaluate):
    """Analyze the accuracy of extracted answers."""
    if not data:
        return 0.0, 0.0

    correct_count = sum(1 for row in data if evaluate(row.get("content", ""), row.get("ref_answer")))
    correct_count_without_context = sum(1 for row in data if evaluate(row.get("content_without_context", ""), row.get("ref_answer")))
    
    total = len(data)
    correct_percentage = (correct_count / total * 100) if total > 0 else 0
    correct_percentage_without_context = (correct_count_without_context / total * 100) if total > 0 else 0

    return correct_percentage, correct_percentage_without_context


def create_stopwords():
    """Create a comprehensive set of stopwords."""
    stopwords = set(corpus.stopwords.words('english'))
    # Single character words
    stopwords.update({chr(i) for i in range(97, 123)})
    # Two character words
    stopwords.update({chr(i) + chr(j) for i in range(97, 123) for j in range(97, 123)})
    # Numbers
    stopwords.update({str(i) for i in range(10)})
    return stopwords

def get_most_common_grams(words, gram=2, n=10):
    """Get most common n-grams from a list of words."""
    if len(words) < gram:
        return []
    grams = ngrams(words, gram)
    gram_counts = Counter(grams)
    # Convert tuple grams to string for JSON serialization
    return [(' '.join(gram), count) for gram, count in gram_counts.most_common(n)]


def analyze_ngrams(data, top_n=10):
    """Analyze and find the most common bigrams and trigrams."""
    # Combine all thinking content into two large strings
    all_thinking_ctx = " ".join([row.get("thinking_content", "") for row in data])
    all_thinking_no_ctx = " ".join([row.get("thinking_content_without_context", "") for row in data])
    
    # Preprocess and tokenize the text
    words_ctx = word_tokenize(preprocess_text(all_thinking_ctx))
    words_no_ctx = word_tokenize(preprocess_text(all_thinking_no_ctx))

    stopwords = create_stopwords()
    # Filter out stopwords
    words_ctx = [word for word in words_ctx if word not in stopwords]
    words_no_ctx = [word for word in words_no_ctx if word not in stopwords]

    # Get most common n-grams
    bigrams_ctx = get_most_common_grams(words_ctx, gram=2, n=top_n)
    bigrams_no_ctx = get_most_common_grams(words_no_ctx, gram=2, n=top_n)
    trigrams_ctx = get_most_common_grams(words_ctx, gram=3, n=top_n)
    trigrams_no_ctx = get_most_common_grams(words_no_ctx, gram=3, n=top_n)
    
    return bigrams_ctx, bigrams_no_ctx, trigrams_ctx, trigrams_no_ctx


def main():
    """Main analysis pipeline."""
    parser = get_args_parser()
    args = parser.parse_args()
    data = load_data(args.model_name, args.data_name, args.results_path)

    if not data:
        print(f"No data to analyze for model: {args.model_name}, dataset: {args.data_name}. Exiting.")
        return {"error": "No data found or loaded"}

    data_generator = get_dataset_generator(args.data_name)
    results = {"model_name": args.model_name, "data_name": args.data_name}

    # Run all analyses and collect results
    (results["avg_thinking_length"], results["avg_thinking_length_without_context"],
     results["thinking_length_t_stat"], results["thinking_length_p_value"]) = analyze_thinking_lengths(data)

    (results["proportion_uncertainty_with_context"], results["proportion_uncertainty_without_context"],
     results["uncertainty_t_stat"], results["uncertainty_p_value"]) = analyze_uncertainty(data)

    (results["abstention_percentage_with_context"],
     results["abstention_percentage_without_context"]) = analyze_abstention(data)

    (results["correct_percentage_with_context"],
     results["correct_percentage_without_context"]) = analyze_answer_accuracy(data, data_generator.evaluate)
    
    (results["answered_percentage_with_context"],
        results["answered_percentage_without_context"]) = analyze_answer_coverage(data, data_generator.extract_answer)
     
    # # Run n-gram analysis
    # (results["top_10_bigrams_with_context"], results["top_10_bigrams_without_context"],
    #  results["top_10_trigrams_with_context"], results["top_10_trigrams_without_context"]) = analyze_ngrams(data)

    # # Convert proportions to percentages for consistency in reporting
    results["proportion_uncertainty_with_context"] *= 100
    results["proportion_uncertainty_without_context"] *= 100

    # save results to a file
    output_path = f"{args.results_path}/{args.model_name}/{args.data_name}_analysis_results.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        # Use indent for readability in the output file
        json.dump(results, f, indent=4)
        
    return results


if __name__ == "__main__":
    # To run this script, you might need to download the 'punkt' tokenizer models for NLTK
    # import nltk
    # nltk.download('punkt')
    analysis_results = main()
    print("\n--- Summary Results ---")
    # Pretty-print the final dictionary to the console
    print(json.dumps(analysis_results, indent=2))