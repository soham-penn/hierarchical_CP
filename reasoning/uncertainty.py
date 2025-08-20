"""Analysis script for thinking content comparison."""

# import nltk
# nltk.download('stopwords')

import argparse
import re
from scipy.stats import ttest_rel
import numpy as np
import os
import json
from utils import load_data

# Ensure these imports are correct based on your project structure
from detect.uncertainty_keywords import *

def get_args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-0.6B')
    parser.add_argument('--data_name', type=str, default='gpqa',
                        help='Choose among: gpqa, gsm8k, mmlu, icraft, imedqa')
    parser.add_argument('--results_path', type=str, default='results',
                        help='Path to save the results')
    return parser


def preprocess_text(text):
    """Remove punctuation, extra whitespace, and convert to lowercase."""
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def count_uncertainty(text, uncertainty_phrases):
    """Count the occurrences of uncertainty phrases in the text."""
    count = 0
    # Ensure text is lowercase for matching
    text_lower = text.lower()
    for phrase in uncertainty_phrases:
        count += text_lower.count(phrase)
    return count


def analyze_uncertainty_by_category(data, uncertainty_keywords_dict):
    """
    Analyze uncertainty phrases by category and perform statistical testing.
    Returns a dictionary of results for each category.
    """
    category_results = {}

    for category, phrases in uncertainty_keywords_dict.items():
        proportions_ctx = []
        proportions_no_ctx = []

        for row in data:
            thinking_content = row.get("thinking_content", "")
            words_ctx = thinking_content.split()
            if words_ctx:
                proportions_ctx.append(count_uncertainty(thinking_content, phrases) / len(words_ctx))
            else:
                proportions_ctx.append(0)

            thinking_content_no_ctx = row.get("thinking_content_without_context", "")
            words_no_ctx = thinking_content_no_ctx.split()
            if words_no_ctx:
                proportions_no_ctx.append(count_uncertainty(thinking_content_no_ctx, phrases) / len(words_no_ctx))
            else:
                proportions_no_ctx.append(0)

        t_stat, p_value = None, None
        # Only perform t-test if there's enough data and lists are of equal length
        if len(proportions_ctx) > 1 and len(proportions_ctx) == len(proportions_no_ctx):
            # Convert to numpy arrays for ttest_rel, handling cases where all values are the same
            # ttest_rel requires at least 2 data points that are not all identical
            if np.std(proportions_ctx) > 1e-9 or np.std(proportions_no_ctx) > 1e-9:
                t_stat, p_value = ttest_rel(proportions_ctx, proportions_no_ctx)
            else: # All values are the same, t-statistic is undefined, p-value is 1 if means are equal, 0 otherwise
                if np.mean(proportions_ctx) == np.mean(proportions_no_ctx):
                    t_stat, p_value = 0.0, 1.0 # Or None, None as appropriate for "undefined"
                else:
                    t_stat, p_value = np.inf, 0.0 # Means are different but std dev is zero, implying infinite t-stat

        avg_proportion_ctx = np.mean(proportions_ctx) if proportions_ctx else 0
        avg_proportion_no_ctx = np.mean(proportions_no_ctx) if proportions_no_ctx else 0

        category_results[category] = {
            "proportion_with_context": avg_proportion_ctx * 100,
            "proportion_without_context": avg_proportion_no_ctx * 100,
            "t_stat": t_stat,
            "p_value": p_value
        }
    return category_results



def main():
    """Main analysis pipeline."""
    parser = get_args_parser()
    args = parser.parse_args()
    data = load_data(args.model_name, args.data_name)
    results_path = args.results_path

    output_path = f"{results_path}/{args.model_name}/{args.data_name}_uncertainty_results.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # skip if the file already exists
    if os.path.exists(output_path):
        print(f"Results file already exists: {output_path}. Skipping analysis.")
        return {"message": "Results file already exists, skipping analysis"}

    if not data:
        print(f"No data to analyze for model: {args.model_name}, dataset: {args.data_name}. Exiting.")
        return {"error": "No data found or loaded"}

    results = {"model_name": args.model_name, "data_name": args.data_name}


    # Analyze each category of uncertainty keywords separately
    category_uncertainty_results = analyze_uncertainty_by_category(data, uncertainty_keywords_dict)
    results["category_uncertainty_results"] = category_uncertainty_results

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