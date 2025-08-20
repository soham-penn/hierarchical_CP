import os
import json
from collections import defaultdict
import datetime

# Define the models and datasets you used in your main.py simulation
MODELS = {
    "RL-Tuned Models": {
        "microsoft/Phi-4-reasoning": {"size": "14B", "rl_tuned": True, "family": "Phi"},
        "Qwen/QwQ-32B": {"size": "32B", "rl_tuned": True, "family": "Qwen"},
        "Qwen/Qwen3-32B": {"size": "32B", "rl_tuned": True, "family": "Qwen"},
    },
    "Distilled Models": {
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {"size": "8B", "rl_tuned": False, "family": "DeepSeek"},
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {"size": "1.5B", "rl_tuned": False, "family": "DeepSeek"},
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {"size": "7B", "rl_tuned": False, "family": "DeepSeek"},
        "Qwen/Qwen3-0.6B": {"size": "0.6B", "rl_tuned": False, "family": "Qwen"},
        "Qwen/Qwen3-1.7B": {"size": "1.7B", "rl_tuned": False, "family": "Qwen"},
        "Qwen/Qwen3-8B": {"size": "8B", "rl_tuned": False, "family": "Qwen"},
    }
}

DATASETS = ["gpqa", "gsm8k", "mmlu", "icraft", "imedqa"]

DATASET_THEMES = {
    "gsm8k": "Math",
    "mmlu": "Math",
    "gpqa": "Science",
    "icraft": "Medical",
    "imedqa": "Medical"
}
THEME_ORDER = ["Math", "Science", "Medical"]
ORDERED_DATASETS = sorted(DATASETS, key=lambda ds: (THEME_ORDER.index(DATASET_THEMES.get(ds, "Z")), ds))

def parse_size(size_str):
    """Converts a model size string (e.g., '14B', '1.5B') to a float for sorting."""
    if not isinstance(size_str, str):
        return 0
    size_str = size_str.upper().replace("B", "")
    try:
        return float(size_str)
    except ValueError:
        return 0

def format_as_markdown_table(header, rows, theme_row=None):
    """Formats a header and rows into a Markdown table string."""
    lines = []
    # Add optional theme row
    if theme_row:
        lines.append("| " + " | ".join(theme_row) + " |")
        lines.append("|" + "---|" * len(header))
    
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in rows:
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)

def generate_comparison_report(all_comparison_results):
    """
    Generates a Markdown report comparing accuracy and abstention changes
    between standard and prompt intervention runs, focusing on accuracy with
    context and abstention without context.
    """
    markdown_lines = [
        f"# Accuracy and Abstention Comparison Report",
        f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    markdown_lines.extend(["\n---", "## Summary of Changes (Standard vs. Prompt Intervention)"])

    models_by_family = defaultdict(list)
    for model_group_name, models_in_group in MODELS.items():
        for model_name, characteristics in models_in_group.items():
            models_by_family[characteristics['family']].append({"model_name": model_name, **characteristics})

    for family, models in sorted(models_by_family.items()):
        markdown_lines.append(f"\n### Family: {family}")
        
        sorted_models = sorted(models, key=lambda m: parse_size(m.get('size')))

        # Table for Accuracy Decrease (with context)
        accuracy_decrease_header = ["Model", "Size"] + [f"{d}" for d in ORDERED_DATASETS]
        accuracy_decrease_theme = ["", ""] + [DATASET_THEMES.get(ds, "") for ds in ORDERED_DATASETS]
        accuracy_decrease_rows = []

        # Table for Abstention Increase (without context)
        abstention_increase_header = ["Model", "Size"] + [f"{d}" for d in ORDERED_DATASETS]
        abstention_increase_theme = ["", ""] + [DATASET_THEMES.get(ds, "") for ds in ORDERED_DATASETS]
        abstention_increase_rows = []

        for model_meta in sorted_models:
            model_name = model_meta['model_name']
            
            accuracy_row = [model_name, model_meta.get('size', 'N/A')]
            abstention_row = [model_name, model_meta.get('size', 'N/A')]

            for ds in ORDERED_DATASETS:
                results = all_comparison_results.get(model_name, {}).get(ds, {})
                
                # Accuracy Decrease (with context)
                accuracy_decrease_with_context = results.get('accuracy_decrease_with_context', 0.0)
                accuracy_row.append(f"{accuracy_decrease_with_context:.2f}%")
                
                # Abstention Increase (without context)
                abstention_increase_without_context = results.get('abstention_increase_without_context', 0.0)
                abstention_row.append(f"{abstention_increase_without_context:.2f}%")
            
            accuracy_decrease_rows.append(accuracy_row)
            abstention_increase_rows.append(abstention_row)
        
        # Add Accuracy Decrease table
        markdown_lines.append("\n**Accuracy Decrease (With Context)**")
        markdown_lines.append(format_as_markdown_table(accuracy_decrease_header, accuracy_decrease_rows, theme_row=accuracy_decrease_theme))
        markdown_lines.append("_Shows the percentage point decrease in accuracy when using the prompt intervention compared to the standard prompt, specifically for generations where context was available._")

        # Add Abstention Increase table
        markdown_lines.append("\n**Abstention Increase (Without Context)**")
        markdown_lines.append(format_as_markdown_table(abstention_increase_header, abstention_increase_rows, theme_row=abstention_increase_theme))
        markdown_lines.append("_Shows the percentage point increase in abstention responses when using the prompt intervention compared to the standard prompt, specifically for generations without context._")

    return "\n".join(markdown_lines)

def main_report_generation():
    all_comparison_results = defaultdict(lambda: defaultdict(dict))
    flat_model_list = {name: chars for _, type_models in MODELS.items() for name, chars in type_models.items()}

    print("Gathering comparison results...")
    for model_name in flat_model_list.keys():
        for data_name in DATASETS:
            results_standard_path = f"results/{model_name}/{data_name}_analysis_results.jsonl"
            results_intervention_path = f"results_prompt_intervention/{model_name}/{data_name}_analysis_results.jsonl"
            
            results_standard = None
            results_intervention = None

            if os.path.exists(results_standard_path):
                with open(results_standard_path, 'r') as f:
                    try:
                        results_standard = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"Error loading standard results for {model_name} on {data_name}: {e}")

            if os.path.exists(results_intervention_path):
                with open(results_intervention_path, 'r') as f:
                    try:
                        results_intervention = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"Error loading intervention results for {model_name} on {data_name}: {e}")

            if results_standard and results_intervention:
                # Perform the comparison, focusing on the specified metrics
                accuracy_decrease_ctx = results_standard.get("correct_percentage_with_context", 0.0) - results_intervention.get("correct_percentage_with_context", 0.0)
                abstention_increase_no_ctx = results_intervention.get("abstention_percentage_without_context", 0.0) - results_standard.get("abstention_percentage_without_context", 0.0)

                all_comparison_results[model_name][data_name] = {
                    "accuracy_decrease_with_context": accuracy_decrease_ctx,
                    "abstention_increase_without_context": abstention_increase_no_ctx,
                }
                print(f"Gathered comparison data for {model_name} on {data_name}")
            else:
                print(f"Missing results for {model_name} on {data_name}. Skipping.")

    print("\nGenerating comparison report...")
    markdown_report = generate_comparison_report(all_comparison_results)
    report_filename = f"prompt_intervention_accuracy_abstention_comparison_report.md" 
    
    try:
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(markdown_report)
        print(f"\n--- Report Generation Complete ---")
        print(f"Comparison report saved to: {report_filename}")
    except IOError as e:
        print(f"\nERROR: Could not write report to file: {e}")

if __name__ == "__main__":
    main_report_generation()