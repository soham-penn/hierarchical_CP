import sys
import os
from collections import defaultdict
import datetime
import statistics
import json
import re

# Add the directory containing eda.py to the Python path
# This assumes eda.py is in the same directory as this script.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import eda
except ImportError:
    print("ERROR: eda.py not found. Make sure it's in the same directory as this script.")
    sys.exit(1)

# Define the models with their characteristics
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


DATASETS = ["gpqa", "gsm8k", "mmlu", "icraft", "imedqa"] # Example datasets
DATASET_THEMES = {
        "gsm8k": "Math",
        "mmlu": "Math",
        "gpqa": "Science",
        "icraft": "Medical",
        "imedqa": "Medical"
    }
THEME_ORDER = ["Math", "Science", "Medical"]


def run_eda_analysis_programmatically(model_name, data_name, results_path='results'):
    """
    Checks for existing results file first. If found, loads it.
    Otherwise, calls eda.main() programmatically to generate it.
    """
    results_path = f"{results_path}/{model_name}/{data_name}_analysis_results.jsonl"
    if os.path.exists(results_path):
        print(f"    -> Found cached results. Loading from {results_path}")
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"    -> WARNING: Could not read cache file {results_path}: {e}. Re-running analysis.")

    print(f"    -> No cache found. Running analysis for {model_name} on {data_name}...")
    original_argv = sys.argv
    sys.argv = ['eda.py', '--model_name', model_name, '--data_name', data_name]
    original_stdout = sys.stdout
    # Suppress stdout from the eda.py script for a cleaner log
    with open(os.devnull, 'w') as null_device:
        sys.stdout = null_device
        results = None
        try:
            results = eda.main()
        except Exception as e:
            # Ensure stdout is restored before printing the error
            sys.stdout = original_stdout
            print(f"ERROR: Failed to run EDA for {model_name} on {data_name}: {e}")
            results = {"error": str(e)}
        finally:
            # Restore stdout
            sys.stdout = original_stdout
            sys.argv = original_argv
    return results

def format_as_markdown_table(header, rows, theme=None):
    """Formats a header and rows into a Markdown table string."""
    if theme is not None:
        lines = [
            "| " + " | ".join(header) + " |",
            "|" + "---|" * len(header),
            "| " + " | ".join(theme) + " |",
        ]
    else:
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)

def parse_size(size_str):
    """Converts a model size string (e.g., '14B', '1.5B') to a float for sorting."""
    if not isinstance(size_str, str):
        return 0
    size_str = size_str.upper().replace("B", "")
    try:
        return float(size_str)
    except ValueError:
        return 0

def generate_markdown_report(all_model_results):
    """
    Generates an analytical Markdown report addressing the key questions.
    """
    markdown_lines = [f"# Model Behavior Analysis Report"]

    # --- Define dataset themes and order ---
    DATASET_THEMES = {
        "gsm8k": "Math", "mmlu": "Math", "gpqa": "Science",
        "icraft": "Medical", "imedqa": "Medical"
    }
    THEME_ORDER = ["Math", "Science", "Medical"]
    ORDERED_DATASETS = sorted(DATASETS, key=lambda ds: (THEME_ORDER.index(DATASET_THEMES.get(ds, "Z")), ds))


    # --- Helper function to check for hypothesis rejection ---
    def check_hypothesis(metrics, p_val_key, val_no_ctx_key, val_ctx_key):
        p_value = metrics.get(p_val_key)
        val_no_ctx = metrics.get(val_no_ctx_key, 0)
        val_ctx = metrics.get(val_ctx_key, 0)
        if p_value is not None and p_value < 0.05 and val_no_ctx > val_ctx:
            return True
        return False

    # --- 1. Executive Summary: Comparison of Model Types by Theme ---
    markdown_lines.extend(["\n---", "## Executive Summary: Comparison of Model Types by Theme"])
    themed_type_stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "think_reject": 0, "uncert_reject": 0}))
    for model, datasets in all_model_results.items():
        for data, metrics in datasets.items():
            if "error" in metrics or data not in DATASET_THEMES:
                continue
            model_type = "RL-Tuned" if metrics.get('rl_tuned') else "Distilled"
            theme = DATASET_THEMES[data]
            
            themed_type_stats[model_type][theme]["total"] += 1
            if check_hypothesis(metrics, 'thinking_length_p_value', 'avg_thinking_length_without_context', 'avg_thinking_length'):
                themed_type_stats[model_type][theme]["think_reject"] += 1
            if check_hypothesis(metrics, 'uncertainty_p_value', 'proportion_uncertainty_without_context', 'proportion_uncertainty_with_context'):
                themed_type_stats[model_type][theme]["uncert_reject"] += 1
    summary_header = ["Model Type", "Theme", "Total Tests", "Thinking Time Hypothesis Rejection Rate", "Uncertainty Hypothesis Rejection Rate"]
    summary_rows = []
    for model_type in ["RL-Tuned", "Distilled"]:
        if model_type in themed_type_stats:
            for theme in THEME_ORDER:
                if theme in themed_type_stats[model_type]:
                    stats = themed_type_stats[model_type][theme]
                    total = stats['total']
                    if total == 0: continue
                    think_rate = f"{stats['think_reject']} / {total} ({stats['think_reject']/total:.1%})"
                    uncert_rate = f"{stats['uncert_reject']} / {total} ({stats['uncert_reject']/total:.1%})"
                    summary_rows.append([model_type, theme, total, think_rate, uncert_rate])
    markdown_lines.append(format_as_markdown_table(summary_header, summary_rows))
    markdown_lines.append("\nThis summary shows the rate at which models reject the null hypothesis, suggesting they 'think' longer or express more uncertainty when given incomplete information, grouped by dataset theme.")

    # --- 2. Hypothesis Rejection Analysis ---
    markdown_lines.extend(["\n---", "## Hypothesis Rejection Analysis"])
    def generate_rejection_section(title, hypothesis, p_key, no_ctx_key, ctx_key):
        markdown_lines.append(f"\n### {title}")
        markdown_lines.append(f"**Hypothesis:** {hypothesis}")
        rejecting_pairs = []
        for model, datasets in all_model_results.items():
            for data, metrics in datasets.items():
                if "error" in metrics: continue
                if check_hypothesis(metrics, p_key, no_ctx_key, ctx_key):
                    rejecting_pairs.append({
                        "model": model, "dataset": data,
                        "no_ctx_val": f"{metrics.get(no_ctx_key, 0):.2f}",
                        "ctx_val": f"{metrics.get(ctx_key, 0):.2f}",
                        "p_value": f"{metrics.get(p_key):.3f}",
                    })
        if rejecting_pairs:
            markdown_lines.append(f"**Result:** The hypothesis was **SUPPORTED** in **{len(rejecting_pairs)}** model-dataset pair(s).")
            header = ["Model", "Dataset", f"Value (No Ctx)", f"Value (Ctx)", "P-Value"]
            rows = [[r["model"], r["dataset"], r["no_ctx_val"], r["ctx_val"], r["p_value"]] for r in rejecting_pairs]
            markdown_lines.append(format_as_markdown_table(header, rows))
        else:
            markdown_lines.append("**Result:** The hypothesis was **NOT SUPPORTED** in any of the tested pairs.")
    generate_rejection_section(
        "Thinking Time Analysis", "Models 'think' longer (higher token length) without full context.",
        'thinking_length_p_value', 'avg_thinking_length_without_context', 'avg_thinking_length'
    )
    generate_rejection_section(
        "Uncertainty Analysis", "Models express more uncertainty without full context.",
        'uncertainty_p_value', 'proportion_uncertainty_without_context', 'proportion_uncertainty_with_context'
    )
    
    # --- 3. Model Performance: Accuracy and Abstention ---
    markdown_lines.extend(["\n---", "## Model Performance: Accuracy and Abstention"])
    
    models_by_family = defaultdict(list)
    for model_name, characteristics in {name: chars for _, type_models in MODELS.items() for name, chars in type_models.items()}.items():
            models_by_family[characteristics['family']].append({"model_name": model_name, **characteristics})

    for family, models in sorted(models_by_family.items()):
        markdown_lines.append(f"\n### Family: {family}")
        
        sorted_models = sorted(models, key=lambda m: parse_size(m.get('size')))

        # Table 1: Coverage
        coverage_header = ["Model", "Size"] + [f"{d}" for d in ORDERED_DATASETS]
        coverage_theme = ["", ""] + [DATASET_THEMES.get(ds, "") for ds in ORDERED_DATASETS]
        coverage_rows = []

        # Table 2: Accuracy
        acc_header = ["Model", "Size"] + [f"{d}" for d in ORDERED_DATASETS]
        acc_theme = ["", ""] + [DATASET_THEMES.get(ds, "") for ds in ORDERED_DATASETS]
        acc_rows = []

        # Table 3: Abstention
        abs_header = ["Model", "Size"] + [f"{d}" for d in ORDERED_DATASETS]
        abs_theme = ["", ""] + [DATASET_THEMES.get(ds, "") for ds in ORDERED_DATASETS]
        abs_rows = []

        for model_meta in sorted_models:
            model_name = model_meta['model_name']
            
            # Populate rows for both tables in one go
            coverage_row = [model_name, model_meta.get('size', 'N/A')]
            acc_row = [model_name, model_meta.get('size', 'N/A')]
            abs_row = [model_name, model_meta.get('size', 'N/A')]
            
            for ds in ORDERED_DATASETS:
                metrics = all_model_results.get(model_name, {}).get(ds, {})
                # Coverage data
                coverage_ctx = metrics.get('answered_percentage_with_context', 0.0)
                coverage_no_ctx = metrics.get('answered_percentage_without_context', 0.0)
                coverage_row.append(f"{coverage_ctx:.1f} / {coverage_no_ctx:.1f}")
                
                # Accuracy data
                acc_ctx = metrics.get('correct_percentage_with_context', 0.0)
                acc_no_ctx = metrics.get('correct_percentage_without_context', 0.0)
                acc_row.append(f"{acc_ctx:.1f} / {acc_no_ctx:.1f}")
                # Abstention data
                abs_ctx = metrics.get('abstention_percentage_with_context', 0.0)
                abs_no_ctx = metrics.get('abstention_percentage_without_context', 0.0)
                abs_row.append(f"{abs_ctx:.1f} / {abs_no_ctx:.1f}")
            
            coverage_rows.append(coverage_row)
            acc_rows.append(acc_row)
            abs_rows.append(abs_row)
        
        # Format and add Coverage table
        markdown_lines.append("\n**Answer Coverage Rates (With Context / Without Context)**")
        markdown_lines.append(format_as_markdown_table(coverage_header, coverage_rows, theme=coverage_theme))
        markdown_lines.append("_Shows how often the model answers questions with and without context (i.e. put a format-wise valid answer in `\\boxed{}`). The ideal result is high coverage with context and low coverage without it._")
        

        # Format and add Accuracy table
        markdown_lines.append("\n**Accuracy Rates (With Context / Without Context)**")
        markdown_lines.append(format_as_markdown_table(acc_header, acc_rows, theme=acc_theme))
        markdown_lines.append("_Shows model correctness. The ideal result is high accuracy with context and low accuracy without it._")

        # Format and add Abstention table
        markdown_lines.append("\n**Abstention Rates (With Context / Without Context)**")
        markdown_lines.append(format_as_markdown_table(abs_header, abs_rows, theme=abs_theme))
        markdown_lines.append("_Shows how often the model abstains. The ideal result is a low abstention rate with context and a high rate without it._")


    # --- 4. Detailed Hypothesis Analysis by Model Family and Size ---
    markdown_lines.extend(["\n---", "## Detailed Hypothesis Analysis by Model Family and Size"])
    for family, models in sorted(models_by_family.items()):
        markdown_lines.append(f"\n### Family: {family}")
        
        sorted_models = sorted(models, key=lambda m: parse_size(m.get('size')))
        
        header = ["Model", "Size"] + \
                     [f"{d} (Len)" for d in ORDERED_DATASETS] + \
                     [f"{d} (Unc)" for d in ORDERED_DATASETS]
        theme = ["", ""] + \
                [DATASET_THEMES.get(ds, "") for ds in ORDERED_DATASETS] + \
                [DATASET_THEMES.get(ds, "") for ds in ORDERED_DATASETS]
        rows = []
        for model_meta in sorted_models:
            model_name = model_meta['model_name']
            row = [model_name, model_meta.get('size', 'N/A')]
            # Check thinking hypothesis
            for ds in ORDERED_DATASETS:
                metrics = all_model_results.get(model_name, {}).get(ds, {})
                row.append("✔️" if check_hypothesis(metrics, 'thinking_length_p_value', 'avg_thinking_length_without_context', 'avg_thinking_length') else "❌")
            # Check uncertainty hypothesis
            for ds in ORDERED_DATASETS:
                metrics = all_model_results.get(model_name, {}).get(ds, {})
                row.append("✔️" if check_hypothesis(metrics, 'uncertainty_p_value', 'proportion_uncertainty_without_context', 'proportion_uncertainty_with_context') else "❌")
            rows.append(row)
        
        markdown_lines.append(format_as_markdown_table(header, rows, theme=theme))
        markdown_lines.append("_✔️ = Hypothesis Supported (p < 0.05 and directional), ❌ = Not Supported. Columns are grouped by theme: Math (gsm8k, mmlu), Science (gpqa), and Medical (icraft, imedqa). (Len) = Thinking Length, (Unc) = Uncertainty._")

    return "\n".join(markdown_lines)


def main():
    """Main comparison pipeline."""
    all_model_results = defaultdict(lambda: defaultdict(dict))
    print(f"Starting model comparison at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S EDT')}...")
    
    flat_model_list = {name: chars for _, type_models in MODELS.items() for name, chars in type_models.items()}
    total_runs = len(flat_model_list) * len(DATASETS)
    current_run = 0

    for model_name, characteristics in flat_model_list.items():
        for data_name in DATASETS:
            current_run += 1
            print(f"({current_run}/{total_runs}) Analyzing {model_name} on {data_name}...")
            
            results = run_eda_analysis_programmatically(model_name, data_name, results_path='results')
            
            # Combine results with model characteristics for reporting
            full_results = {**results, **characteristics, "model_name": model_name, "dataset_name": data_name}
            all_model_results[model_name][data_name] = full_results

    print("\nGenerating analytical report...")
    markdown_report = generate_markdown_report(all_model_results)
    report_filename = f"analytical_report.md"
    
    try:
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(markdown_report)
        print(f"\n--- Analysis Complete ---")
        print(f"Analytical report saved to: {report_filename}")
    except IOError as e:
        print(f"\nERROR: Could not write report to file: {e}")

if __name__ == "__main__":
    main()