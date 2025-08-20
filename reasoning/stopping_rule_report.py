import os
import json
import csv
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description="Process stopping rule results.")
parser.add_argument("--results_path", type=str, default="results", help="Directory containing the results files.")
args = parser.parse_args()

# Define the models and datasets you used in your main.py simulation
MODELS = {
    "RL-Tuned Models": {
        "Qwen/QwQ-32B": {"size": "32B", "rl_tuned": True, "family": "Qwen"},
        "Qwen/Qwen3-32B": {"size": "32B", "rl_tuned": True, "family": "Qwen"},
        "microsoft/Phi-4-reasoning-plus": {"size": "14B", "rl_tuned": True, "family": "Phi"},
        "nvidia/AceReason-Nemotron-7B": {"size": "7B", "rl_tuned": True, "family": "Nemotron"},
        "nvidia/AceReason-Nemotron-1.1-7B": {"size": "7B", "rl_tuned": True, "family": "Nemotron"},
        "nvidia/AceReason-Nemotron-14B": {"size": "14B", "rl_tuned": True, "family": "Nemotron"},
        "XiaomiMiMo/MiMo-7B-RL-0530": {"size": "7B", "rl_tuned": True, "family": "MiMo"},
        "Skywork/Skywork-OR1-7B": {"size": "7B", "rl_tuned": True, "family": "Skywork"},
        "Skywork/Skywork-OR1-32B": {"size": "32B", "rl_tuned": True, "family": "Skywork"}
    },
    "Distilled Models": {
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {"size": "8B", "rl_tuned": False, "family": "DeepSeek"},
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {"size": "1.5B", "rl_tuned": False, "family": "DeepSeek"},
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {"size": "7B", "rl_tuned": False, "family": "DeepSeek"},
        "Qwen/Qwen3-0.6B": {"size": "0.6B", "rl_tuned": False, "family": "Qwen"},
        "Qwen/Qwen3-1.7B": {"size": "1.7B", "rl_tuned": False, "family": "Qwen"},
        "Qwen/Qwen3-8B": {"size": "8B", "rl_tuned": False, "family": "Qwen"},
        "microsoft/Phi-4-reasoning": {"size": "14B", "rl_tuned": False, "family": "Phi"},
        "microsoft/Phi-4-mini-reasoning": {"size": "4B", "rl_tuned": False, "family": "Phi"},
        "microsoft/Phi-4-mini-flash-reasoning": {"size": "4B", "rl_tuned": False, "family": "Phi"}
    }
}

DATASETS = ["gpqa", "gsm8k", "mmlu", "icraft", "imedqa"]
DATASET_THEMES = {
    "gsm8k": "Math", "mmlu": "Math", "gpqa": "Science",
    "icraft": "Medical", "imedqa": "Medical"
}
THEME_ORDER = ["Math", "Science", "Medical"]
ORDERED_DATASETS = sorted(DATASETS, key=lambda ds: (THEME_ORDER.index(DATASET_THEMES.get(ds, "Z")), ds))

def format_as_markdown_table(header, rows, theme_row=None):
    """Formats a list of lists into a Markdown table."""
    lines = []
    if theme_row:
        lines.append("| " + " | ".join(theme_row) + " |")
        lines.append("|" + "---|" * len(header))
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in rows:
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)

def parse_size(size_str):
    """Parses a model size string (e.g., '7B') into a float."""
    if not isinstance(size_str, str): return 0
    size_str = size_str.upper().replace("B", "")
    try: return float(size_str)
    except ValueError: return 0

def save_results_to_files(processed_results, base_filename):
    """Saves the structured results to JSON and CSV files."""
    
    # --- Save to JSON ---
    json_filename = f"{base_filename}.json"
    try:
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(processed_results, f, indent=2)
        print(f"Aggregated results saved to: {json_filename}")
    except IOError as e:
        print(f"\nERROR: Could not write JSON to file: {e}")

    # --- Save to CSV ---
    csv_filename = f"{base_filename}.csv"
    headers = [
        "model_group", "model_name", "model_size", "family", "dataset",
        "stopping_rule", "quantile",
        "avg_tokens_saved_with_context", "avg_tokens_saved_without_context",
        "avg_percentage_saved_with_context", "avg_percentage_saved_without_context",
        "early_stopping_rate_with_context", "early_stopping_rate_without_context",
        "accuracy_dropped_with_context", "abstention_increased_without_context"
    ]
    
    try:
        with open(csv_filename, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for group_name, data in processed_results.items():
                for model in data["models"]:
                    for ds, rule_results in model["metrics"].items():
                        for rule_name, quantile_results in rule_results.items():
                            for quantile, metrics in quantile_results.items():
                                # Create a copy and remove the 'best_quantile' key if it exists
                                metrics_copy = metrics.copy()
                                metrics_copy.pop('best_quantile', None)

                                row = {
                                    "model_group": group_name,
                                    "model_name": model["model_name"],
                                    "model_size": model["size"],
                                    "family": model["family"],
                                    "dataset": ds,
                                    "stopping_rule": rule_name,
                                    "quantile": quantile,
                                    **metrics_copy
                                }
                                writer.writerow(row)
        print(f"Aggregated results saved to: {csv_filename}")
    except IOError as e:
        print(f"\nERROR: Could not write CSV to file: {e}")

def get_first_metrics_for_rule(rule_results):
    """
    Returns the metrics for the first available quantile,
    effectively ignoring the quantile selection logic.
    """
    if rule_results:
        # Return the first value from the dictionary
        first_quantile = next(iter(rule_results))
        return rule_results[first_quantile]
    return {}

def generate_stopping_rule_report(processed_results):
    """Generates a Markdown report from pre-processed results, split by stopping rule type."""
    markdown_lines = ["# Stopping Rule Effectiveness Report"]

    stopping_rules_to_report = ["LengthStoppingRule", "UncertaintyStoppingRule"]

    for model_group_name, data in processed_results.items():
        markdown_lines.append(f"\n---\n## {model_group_name}")

        sorted_models = data["models"]

        for rule_type in stopping_rules_to_report:
            markdown_lines.append(f"\n### Results for {rule_type}")
            markdown_lines.append(f"*(Data represents the first available quantile for each model, as quantile information is ignored.)*")

            # --- Tables for 'With Context' Metrics ---
            markdown_lines.append("\n#### With Context")
            
            # Metric headers and corresponding dictionary keys
            metrics_to_report = [
                ("Average Tokens Saved", 'avg_tokens_saved_with_context'),
                ("Average Percentage of Tokens Saved", 'avg_percentage_saved_with_context'),
                ("Early Stopping Rate", 'early_stopping_rate_with_context'),
                ("Accuracy Dropped", 'accuracy_dropped_with_context'),
            ]

            for table_title, metric_key in metrics_to_report:
                header = ["Model", "Size"] + [f"{d}" for d in ORDERED_DATASETS]
                theme_row = ["", ""] + [DATASET_THEMES.get(ds, "") for ds in ORDERED_DATASETS]
                rows = []
                
                group_total = defaultdict(float)
                model_count = defaultdict(int)

                for model in sorted_models:
                    row_data = [model["model_name"], model["size"]]
                    for ds in ORDERED_DATASETS:
                        ds_results = model["metrics"].get(ds, {})
                        rule_results = ds_results.get(rule_type, {})
                        
                        if not rule_results:
                            row_data.append("N/A")
                        else:
                            metrics = get_first_metrics_for_rule(rule_results)
                            value = metrics.get(metric_key, 0)
                            
                            row_data.append(f"{value:.2f}{'%' if 'percentage' in metric_key or 'rate' in metric_key or 'dropped' in metric_key else ''}")
                            
                            group_total[ds] += value
                            model_count[ds] += 1
                    rows.append(row_data)
                
                # Add average row
                avg_row = ['**Average**', '']
                for ds in ORDERED_DATASETS:
                    count = model_count[ds]
                    avg_value = group_total[ds] / count if count > 0 else 0
                    avg_row.append(f"{avg_value:.2f}{'%' if 'percentage' in metric_key or 'rate' in metric_key or 'dropped' in metric_key else ''}")
                rows.append(avg_row)
                
                markdown_lines.append(f"\n**{table_title}**")
                markdown_lines.append(format_as_markdown_table(header, rows, theme_row=theme_row))

            # --- Tables for 'Without Context' Metrics ---
            markdown_lines.append("\n#### Without Context")

            metrics_to_report_without_context = [
                ("Average Tokens Saved", 'avg_tokens_saved_without_context'),
                ("Average Percentage of Tokens Saved", 'avg_percentage_saved_without_context'),
                ("Early Stopping Rate", 'early_stopping_rate_without_context'),
                ("Abstention Increased", 'abstention_increased_without_context'),
            ]
            
            for table_title, metric_key in metrics_to_report_without_context:
                header = ["Model", "Size"] + [f"{d}" for d in ORDERED_DATASETS]
                theme_row = ["", ""] + [DATASET_THEMES.get(ds, "") for ds in ORDERED_DATASETS]
                rows = []

                group_total = defaultdict(float)
                model_count = defaultdict(int)

                for model in sorted_models:
                    row_data = [model["model_name"], model["size"]]
                    for ds in ORDERED_DATASETS:
                        ds_results = model["metrics"].get(ds, {})
                        rule_results = ds_results.get(rule_type, {})
                        
                        if not rule_results:
                            row_data.append("N/A")
                        else:
                            metrics = get_first_metrics_for_rule(rule_results)
                            value = metrics.get(metric_key, 0)
                            
                            row_data.append(f"{value:.2f}{'%' if 'percentage' in metric_key or 'rate' in metric_key or 'increased' in metric_key else ''}")
                            
                            group_total[ds] += value
                            model_count[ds] += 1
                    rows.append(row_data)

                # Add average row
                avg_row = ['**Average**', '']
                for ds in ORDERED_DATASETS:
                    count = model_count[ds]
                    avg_value = group_total[ds] / count if count > 0 else 0
                    avg_row.append(f"{avg_value:.2f}{'%' if 'percentage' in metric_key or 'rate' in metric_key or 'increased' in metric_key else ''}")
                rows.append(avg_row)

                markdown_lines.append(f"\n**{table_title}**")
                markdown_lines.append(format_as_markdown_table(header, rows, theme_row=theme_row))
    
    # New section for final comparison
    markdown_lines.append("\n---\n## Final Comparison: LengthStoppingRule vs. UncertaintyStoppingRule")
    markdown_lines.append("*(Comparing the aggregated averages across all datasets for each model group.)*")

    comparison_metrics_with_context = [
        ("Avg. Tokens Saved (W/ Context)", 'avg_tokens_saved_with_context'),
        ("Avg. % Saved (W/ Context)", 'avg_percentage_saved_with_context'),
        ("Early Stopping Rate (W/ Context)", 'early_stopping_rate_with_context'),
        ("Accuracy Dropped (W/ Context)", 'accuracy_dropped_with_context')
    ]

    comparison_metrics_without_context = [
        ("Avg. Tokens Saved (W/o Context)", 'avg_tokens_saved_without_context'),
        ("Avg. % Saved (W/o Context)", 'avg_percentage_saved_without_context'),
        ("Early Stopping Rate (W/o Context)", 'early_stopping_rate_without_context'),
        ("Abstention Increased (W/o Context)", 'abstention_increased_without_context')
    ]

    for model_group_name, data in processed_results.items():
        markdown_lines.append(f"\n### {model_group_name} Aggregated Averages")
        
        # Calculate group averages for each rule
        group_averages = defaultdict(lambda: defaultdict(float))
        group_counts = defaultdict(lambda: defaultdict(int))
        
        for model in data["models"]:
            for ds in ORDERED_DATASETS:
                for rule_type in stopping_rules_to_report:
                    ds_results = model["metrics"].get(ds, {})
                    rule_results = ds_results.get(rule_type, {})
                    
                    if rule_results:
                        metrics = get_first_metrics_for_rule(rule_results)
                        for _, metric_key in comparison_metrics_with_context + comparison_metrics_without_context:
                            value = metrics.get(metric_key)
                            if value is not None:
                                group_averages[rule_type][metric_key] += value
                                group_counts[rule_type][metric_key] += 1
        
        # Table for With Context
        header_with_context = ["Metric"] + stopping_rules_to_report
        rows_with_context = []
        for title, key in comparison_metrics_with_context:
            row = [title]
            for rule in stopping_rules_to_report:
                count = group_counts[rule][key]
                avg_value = group_averages[rule][key] / count if count > 0 else 0
                row.append(f"{avg_value:.2f}{'%' if 'percentage' in key or 'rate' in key or 'dropped' in key else ''}")
            rows_with_context.append(row)
        markdown_lines.append(format_as_markdown_table(header_with_context, rows_with_context))

        # Table for Without Context
        header_without_context = ["Metric"] + stopping_rules_to_report
        rows_without_context = []
        for title, key in comparison_metrics_without_context:
            row = [title]
            for rule in stopping_rules_to_report:
                count = group_counts[rule][key]
                avg_value = group_averages[rule][key] / count if count > 0 else 0
                row.append(f"{avg_value:.2f}{'%' if 'percentage' in key or 'rate' in key or 'increased' in key else ''}")
            rows_without_context.append(row)
        markdown_lines.append(format_as_markdown_table(header_without_context, rows_without_context))

    return "\n".join(markdown_lines)


def main():
    """Main function to load data, process it, and generate all outputs."""
    all_stopping_rule_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    flat_model_list = {name: chars for _, type_models in MODELS.items() for name, chars in type_models.items()}
    
    print("Gathering stopping rule results...")
    for model_name in flat_model_list.keys():
        for data_name in DATASETS:
            path = f"{args.results_path}/{model_name}/{data_name}_stopping_rule_results.jsonl"
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        for line in f:
                            results = json.loads(line)
                            rule_name = results.get("stopping_rule")
                            quantile = results.get("quantile")
                            if rule_name and quantile:
                                all_stopping_rule_results[model_name][data_name][rule_name][quantile] = results
                            else:
                                print(f"Warning: Skipping malformed line in {path}")
                    print(f"Loaded results for {model_name} on {data_name}")
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Error loading {path}: {e}")
            else:
                print(f"Results file not found: {path}")

    print("\nProcessing results...")
    processed_results = defaultdict(lambda: {"models": [], "averages": {}})
    
    for group_name, models_in_group in MODELS.items():
        model_list = []
        for model_name, characteristics in models_in_group.items():
            model_data = {
                **characteristics,
                "model_name": model_name,
                "metrics": all_stopping_rule_results.get(model_name, {})
            }
            model_list.append(model_data)

        processed_results[group_name]["models"] = sorted(model_list, key=lambda m: parse_size(m.get('size')))

    # --- 3. Generate All Output Files ---
    print("\nGenerating stopping rule effectiveness report...")
    markdown_report = generate_stopping_rule_report(processed_results)
    report_filename = f"{args.results_path}/stopping_rule_effectiveness_report.md"
    try:
        with open(report_filename, "w", encoding="utf-8") as f: f.write(markdown_report)
        print(f"Stopping rule effectiveness report saved to: {report_filename}")
    except IOError as e:
        print(f"\nERROR: Could not write report to file: {e}")

    print("\nSaving aggregated results to JSON and CSV...")
    save_results_to_files(processed_results, f"{args.results_path}/stopping_rule_results_summary")

    print("\n--- All Operations Complete ---")

if __name__ == "__main__":
    main()