# test_poisson_assumptions.py (Now performs a two-part test)

import argparse
import re
import numpy as np
import os
import json
from scipy.stats import kstest, expon
from statsmodels.stats.diagnostic import acorr_ljungbox
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data
from detect.uncertainty_keywords import uncertainty_keywords_dict

def get_args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('Poisson Assumption Test Arguments', add_help=False)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-0.6B')
    parser.add_argument('--data_name', type=str, default='gpqa')
    parser.add_argument('--p_value_threshold', type=float, default=0.05)
    parser.add_argument('--visualize', action='store_true', help='Enable to generate and save the KDE plot')
    return parser

def preprocess_text(text):
    """
    Removes punctuation, extra whitespace, and converts to lowercase.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def find_keyword_positions(text, keywords):
    """
    Finds the starting word indices of all keyword occurrences in a given text.
    """
    positions = []
    clean_text = preprocess_text(text)
    tokens = clean_text.split()
    
    clean_keywords = [preprocess_text(kw) for kw in keywords if kw]
    clean_keywords = [kw for kw in clean_keywords if kw]

    i = 0
    while i < len(tokens):
        match_found = False
        for keyword in clean_keywords:
            keyword_tokens = keyword.split()
            if tokens[i : i + len(keyword_tokens)] == keyword_tokens:
                positions.append(i)
                i += len(keyword_tokens)
                match_found = True
                break
        if not match_found:
            i += 1
            
    return positions

def visualize_kde_subplots(normalized_positions_wc, normalized_positions_woc, category_name, output_file):
    """
    Generates and saves a KDE plot with two subplots for normalized keyword positions.
    """
    # Create the directory for the plot if it doesn't exist.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    fig.suptitle(f'Normalized Keyword Position Density for: {category_name}', fontsize=16)

    # Subplot 1: With Context
    axes[0].set_title('With Context')
    if len(normalized_positions_wc) > 1:
        sns.kdeplot(normalized_positions_wc, fill=True, bw_adjust=0.5, ax=axes[0])
        axes[0].set_xlim(0, 1)
    else:
        axes[0].text(0.5, 0.5, 'Not enough data for KDE plot.', ha='center', transform=axes[0].transAxes)
    axes[0].set_xlabel('Normalized Position within Trace (0=start, 1=end)', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Subplot 2: Without Context
    axes[1].set_title('Without Context')
    if len(normalized_positions_woc) > 1:
        sns.kdeplot(normalized_positions_woc, fill=True, bw_adjust=0.5, ax=axes[1])
        axes[1].set_xlim(0, 1)
    else:
        axes[1].text(0.5, 0.5, 'Not enough data for KDE plot.', ha='center', transform=axes[1].transAxes)
    axes[1].set_xlabel('Normalized Position within Trace (0=start, 1=end)', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file)
    plt.close()
    print(f"-> KDE plot saved to {output_file}")


def visualize_histogram_subplots(intervals_wc, intervals_woc, category_name, output_file):
    """
    Generates and saves a histogram with two subplots for inter-arrival times.
    """
    # Create the directory for the plot if it doesn't exist.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    fig.suptitle(f'Inter-arrival Time Histogram for: {category_name}', fontsize=16)

    # Subplot 1: With Context
    axes[0].set_title('With Context')
    if len(intervals_wc) > 1:
        axes[0].hist(intervals_wc, bins='auto', color='skyblue', alpha=0.7)
        axes[0].grid(True, linestyle='--', alpha=0.6)
    else:
        axes[0].text(0.5, 0.5, 'Not enough data for histogram.', ha='center', transform=axes[0].transAxes)
    axes[0].set_xlabel('Inter-arrival Time (in words)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)

    # Subplot 2: Without Context
    axes[1].set_title('Without Context')
    if len(intervals_woc) > 1:
        axes[1].hist(intervals_woc, bins='auto', color='skyblue', alpha=0.7)
        axes[1].grid(True, linestyle='--', alpha=0.6)
    else:
        axes[1].text(0.5, 0.5, 'Not enough data for histogram.', ha='center', transform=axes[1].transAxes)
    axes[1].set_xlabel('Inter-arrival Time (in words)', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file)
    plt.close()
    print(f"-> Histogram saved to {output_file}")


def test_poisson_assumptions(traces, keywords, p_value_threshold):
    """
    Performs a two-part test for Poisson process assumptions:
    1. K-S test for exponential distribution of inter-arrival times.
    2. Ljung-Box test for independence of inter-arrival times.
    Also returns data for visualization.
    """
    all_intervals = []
    total_event_count = 0
    all_normalized_positions = []

    for trace in traces:
        clean_trace_text = preprocess_text(trace)
        trace_length = len(clean_trace_text.split())
        positions_in_trace = find_keyword_positions(trace, keywords)
        total_event_count += len(positions_in_trace)

        if trace_length > 0 and positions_in_trace:
            normalized_positions = [p / trace_length for p in positions_in_trace]
            all_normalized_positions.extend(normalized_positions)
        
        if positions_in_trace:
            full_positions = np.insert(positions_in_trace, 0, 0)
            intervals_in_trace = np.diff(full_positions)
            all_intervals.extend(intervals_in_trace)

    results_dict = {}
    if len(all_intervals) < 20:
        results_dict = {
            "is_poisson": "N/A (Few Events)",
            "ks_p_value": None, "ljung_box_p_value": None, "event_count": total_event_count
        }
    else:
        _, ks_p_value = kstest(all_intervals, 'expon', args=(0, np.mean(all_intervals)))
        is_exponential = ks_p_value >= p_value_threshold
        lag = max(1, int(np.floor(np.log(len(all_intervals)))))  
        ljung_box_result = acorr_ljungbox(all_intervals, lags=[lag], return_df=True)
        lb_p_value = ljung_box_result['lb_pvalue'].iloc[0]
        is_independent = lb_p_value >= p_value_threshold
        is_poisson = is_exponential and is_independent
        results_dict = {
            "is_poisson": bool(is_poisson),
            "ks_p_value": float(ks_p_value),
            "ljung_box_p_value": float(lb_p_value),
            "event_count": total_event_count,
        }

    return results_dict, all_normalized_positions, all_intervals

def main():
    """Main analysis pipeline for testing Poisson assumptions."""
    parser = get_args_parser()
    args = parser.parse_args()
    
    print(f"--- Testing Poisson Assumptions for {args.model_name} on {args.data_name} ---")
    
    data = load_data(args.model_name, args.data_name)

    if not data:
        print(f"No data found for model: {args.model_name}, dataset: {args.data_name}. Exiting.")
        return

    traces_with_context = [row.get("thinking_content", "") for row in data]
    traces_without_context = [row.get("thinking_content_without_context", "") for row in data]

    results = defaultdict(dict)
    
    sorted_uncertainty_keywords_dict = {
        cat: sorted(kw_list, key=lambda x: len(x.split()), reverse=True)
        for cat, kw_list in uncertainty_keywords_dict.items()
    }

    for category, keywords in sorted_uncertainty_keywords_dict.items():
        print(f"\nAnalyzing Category: '{category}'")
        
        results_with_context, norm_pos_wc, intervals_wc = test_poisson_assumptions(
            traces_with_context, keywords, args.p_value_threshold
        )
        print(f"  With Context: Is Poisson = {results_with_context['is_poisson']}, "
            f"K-S p-val = {results_with_context['ks_p_value']:.4f}" if results_with_context['ks_p_value'] is not None else "K-S p-val = N/A, "
            f"Ljung-Box p-val = {results_with_context['ljung_box_p_value']:.4f}" if results_with_context['ljung_box_p_value'] is not None else "Ljung-Box p-val = N/A")

        results_without_context, norm_pos_woc, intervals_woc = test_poisson_assumptions(
            traces_without_context, keywords, args.p_value_threshold
        )
        print(f"  Without Context: Is Poisson = {results_without_context['is_poisson']}, "
            f"K-S p-val = {results_without_context['ks_p_value']:.4f}" if results_without_context['ks_p_value'] is not None else "K-S p-val = N/A, "
            f"Ljung-Box p-val = {results_without_context['ljung_box_p_value']:.4f}" if results_without_context['ljung_box_p_value'] is not None else "Ljung-Box p-val = N/A")
            
        results[category]['with_context'] = results_with_context
        results[category]['without_context'] = results_without_context

        if args.visualize:
            if norm_pos_wc or norm_pos_woc:
                plot_path_kde = f"results/{args.model_name}/{args.data_name}/{category}_kde.png"
                visualize_kde_subplots(norm_pos_wc, norm_pos_woc, category, plot_path_kde)
            
            if intervals_wc or intervals_woc:
                plot_path_hist = f"results/{args.model_name}/{args.data_name}/{category}_hist.png"
                visualize_histogram_subplots(intervals_wc, intervals_woc, category, plot_path_hist)

    output_path = f"results/{args.model_name}/{args.data_name}_poisson_test_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nAnalysis complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()