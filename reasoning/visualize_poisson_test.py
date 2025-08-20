import os
import json
from collections import defaultdict
import pandas as pd

def get_symbol(is_poisson_result):
    """Converts the 'is_poisson' result to a symbol."""
    if is_poisson_result is True:
        return "✅"
    if is_poisson_result is False:
        return "❌"
    return "⚪️"  # For "N/A (Few Events)"

def collect_and_visualize_poisson_results(models, datasets, results_base_dir="results", report_filename="poisson_test_report.md"):
    """
    Collects Poisson test results, processes them, and saves formatted tables to a report file.

    Args:
        models (list): List of model names.
        datasets (list): List of dataset names.
        results_base_dir (str): Base directory where results are stored.
        report_filename (str): Name of the markdown report file to save.
    """
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    report_content = []

    # Iterate through models and datasets to find result files
    for model in models:
        for dataset in datasets:
            result_file_path = os.path.join(results_base_dir, model, f"{dataset}_poisson_test_results.json")

            if os.path.exists(result_file_path):
                with open(result_file_path, 'r', encoding='utf-8') as f:
                    try:
                        results = json.load(f)
                        for category, category_results in results.items():
                            all_results[category][model][dataset] = category_results
                    except json.JSONDecodeError:
                        report_content.append(f"Error decoding JSON from {result_file_path}\n")
            else:
                report_content.append(f"Result file not found: {result_file_path}\n")

    # Prepare data for tables
    category_tables_data = defaultdict(lambda: defaultdict(dict))

    for category, models_data in all_results.items():
        for model, datasets_data in models_data.items():
            for dataset, results in datasets_data.items():
                with_context_result = results.get("with_context", {}).get("is_poisson")
                without_context_result = results.get("without_context", {}).get("is_poisson")

                symbol_with = get_symbol(with_context_result)
                symbol_without = get_symbol(without_context_result)
                
                category_tables_data[category][(model, dataset)] = f"{symbol_with}/{symbol_without}"

    # Generate and add tables to report content for each category
    report_content.append("# Poisson Process Assumption Test Report\n\n")
    report_content.append("## Legend\n")
    report_content.append("- **✅**: Passes both K-S and Ljung-Box tests (Consistent with Poisson process).\n")
    report_content.append("- **❌**: Fails at least one of the tests (Not consistent with Poisson process).\n")
    report_content.append("- **⚪️**: Not enough events to perform the test (<20 intervals).\n")
    report_content.append("- **Format**: `With Context` / `Without Context`\n\n")

    sorted_categories = sorted(category_tables_data.keys())

    for category in sorted_categories:
        category_data = category_tables_data[category]
        report_content.append(f"## Results for Category: `{category}`\n")
        report_content.append("---\n\n")

        unique_models = sorted(list(set([k[0] for k in category_data.keys()])))
        unique_datasets = sorted(list(set([k[1] for k in category_data.keys()])))
        
        df_rows = []
        for dataset in unique_datasets:
            row_data = {"Dataset": dataset}
            for model in unique_models:
                key = (model, dataset)
                row_data[model] = category_data.get(key, "-/-")
            df_rows.append(row_data)

        if not df_rows:
            report_content.append("No data available for this category.\n\n")
            continue

        df = pd.DataFrame(df_rows)
        
        # Ensure all models are columns, even if some have no data for this category
        all_columns = ["Dataset"] + unique_models
        df = df.reindex(columns=all_columns, fill_value="-/-")

        report_content.append(df.to_markdown(index=False, tablefmt="pipe") + "\n\n")

    # Define the output path for the report file
    report_output_path = os.path.join(results_base_dir, report_filename)
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)

    # Write all collected content to the report file
    with open(report_output_path, 'w', encoding='utf-8') as f:
        f.write("".join(report_content))

    print(f"\nAnalysis report saved to: {report_output_path}")

# Define your models and datasets here
models = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-32B", "Qwen/QwQ-32B", "microsoft/Phi-4-reasoning", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]
datasets = ["gpqa", "gsm8k", "mmlu", "icraft", "imedqa"]

if __name__ == "__main__":
    collect_and_visualize_poisson_results(models, datasets)
