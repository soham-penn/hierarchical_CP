import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Using the same MODELS dictionary to categorize results
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

def load_analysis_data(base_dir):
    """Recursively finds and loads all '*_analysis_results.jsonl' files."""
    all_results = []
    print(f"Loading data from: {base_dir}")
    if not os.path.isdir(base_dir):
        print(f"Warning: Directory not found: {base_dir}")
        return []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_analysis_results.jsonl"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            for line in f:
                                if line.strip():
                                    all_results.append(json.loads(line))
                        except json.JSONDecodeError:
                            f.seek(0)
                            all_results.append(json.load(f))
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    print(f"Found and successfully loaded {len(all_results)} result objects in {base_dir}.\n")
    return all_results

def load_stopping_rule_data(filepath, baseline_df):
    """Loads stopping rule data, calculates absolute values, and returns a DataFrame."""
    print(f"Loading and processing stopping rule data from: {filepath}")
    if not os.path.exists(filepath):
        print(f"Warning: Stopping rule file not found: {filepath}")
        return pd.DataFrame()

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    baseline_lookup = baseline_df.set_index(['model_name', 'data_name']).to_dict('index')
    
    stopping_rule_results = []
    # Loop through each model group and its models
    for group_name, group_data in data.items():
        for model_data in group_data["models"]:
            model_name = model_data["model_name"]
            # Loop through each dataset
            for data_name, metrics_by_rule in model_data["metrics"].items():
                baseline_key = (model_name, data_name)
                if baseline_key not in baseline_lookup:
                    continue

                baseline_metrics = baseline_lookup[baseline_key]
                
                # Loop through each stopping rule found in the JSON
                for rule_name, quantile_results in metrics_by_rule.items():
                    if not quantile_results:
                        continue
                    
                    # We'll take the first quantile's metrics as the representative for the rule
                    first_quantile = next(iter(quantile_results))
                    metrics = quantile_results[first_quantile]

                    absolute_correct_wc = baseline_metrics.get('correct_percentage_with_context', 0) - metrics.get('accuracy_dropped_with_context', 0)
                    absolute_abstention_woc = baseline_metrics.get('abstention_percentage_without_context', 0) + metrics.get('abstention_increased_without_context', 0)
                    
                    stopping_rule_results.append({
                        'model_name': model_name,
                        'data_name': data_name,
                        'correct_percentage_with_context': absolute_correct_wc,
                        'abstention_percentage_without_context': absolute_abstention_woc,
                        'condition': f"Stopping Rule: {rule_name.replace('StoppingRule', '').replace('Length', 'Length ').replace('Uncertainty', 'Uncertainty ').strip()}"
                    })

    print(f"Processed {len(stopping_rule_results)} stopping rule results.\n")
    return pd.DataFrame(stopping_rule_results)

def shorten_model_name(full_name):
    """Shortens 'google/gemma-7b' to 'gemma-7b'."""
    return full_name.split('/')[-1]

def plot_charts(df, plot_title, output_filename):
    """Plots a 2x1 grid of 4-way grouped horizontal bar charts."""
    if df.empty:
        print(f"Skipping plot generation for '{plot_title}' because no data was provided.")
        return

    print(f"Generating comprehensive bar charts for '{plot_title}' and saving to {output_filename}...")

    df['short_name'] = df['model_name'].apply(shorten_model_name)

    sns.set_style("whitegrid")
    num_models = len(df['short_name'].unique())
    fig_height = max(8, num_models * 0.8)
    
    fig, axes = plt.subplots(2, 1, figsize=(18, fig_height))
    fig.suptitle(plot_title, fontsize=22, y=1.0)
    
    palette = {
        "Baseline": "royalblue",
        "Prompt Intervention": "darkorange",
        "Stopping Rule: Length": "mediumseagreen",
        "Stopping Rule: Uncertainty": "purple"
    }

    plot_metrics = {
        ("Correct % (With Context)", "correct_percentage_with_context"): axes[0],
        ("Abstention % (Without Context)", "abstention_percentage_without_context"): axes[1],
    }
    
    # Sort by baseline performance for more insightful plots
    baseline_perf = df[df['condition'] == 'Baseline'].set_index('short_name')['correct_percentage_with_context']
    sorted_model_order = baseline_perf.sort_values(ascending=False).index
    
    df['short_name'] = pd.Categorical(df['short_name'], categories=sorted_model_order, ordered=True)
    df_sorted_for_plot = df.sort_values(['short_name', 'condition'], ascending=[False, True])

    for (title, metric_col), ax in plot_metrics.items():
        data_for_plot = df_sorted_for_plot[['short_name', 'condition', metric_col]].rename(columns={metric_col: 'percentage'})

        sns.barplot(data=data_for_plot, y='short_name', x='percentage', hue='condition', palette=palette, ax=ax, orient='h')
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Percentage (%)", fontsize=12)
        ax.set_ylabel("")
        ax.legend(title='Condition')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.set_xlim(0, 100)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {output_filename}")


def main():
    """Main function to run the comparison and plotting."""
    # Create the figs directory if it doesn't exist
    if not os.path.exists("figs"):
        os.makedirs("figs")

    # --- 1. Load All Data Sources ---
    baseline_df = pd.DataFrame(load_analysis_data('results'))
    prompt_df = pd.DataFrame(load_analysis_data('results_prompt_intervention'))
    stopping_rule_df = load_stopping_rule_data('results/stopping_rule_results_summary.json', baseline_df)

    if baseline_df.empty or prompt_df.empty or stopping_rule_df.empty:
        print("Could not load data from one or more required sources. Exiting.")
        return

    # --- 2. Prepare Data for Combination ---
    cols_to_keep = ['model_name', 'data_name', 'correct_percentage_with_context', 'abstention_percentage_without_context']
    baseline_df = baseline_df[cols_to_keep].copy()
    prompt_df = prompt_df[cols_to_keep].copy()

    baseline_df['condition'] = 'Baseline'
    prompt_df['condition'] = 'Prompt Intervention'

    # --- 3. Combine into a Single DataFrame ---
    comparison_df = pd.concat([baseline_df, prompt_df, stopping_rule_df], ignore_index=True)

    # --- 4. Classify Models and Plot ---
    model_to_group = {
        model_name: group_name
        for group_name, models_in_group in MODELS.items()
        for model_name in models_in_group.keys()
    }
    comparison_df['model_group'] = comparison_df['model_name'].map(model_to_group)
    
    print("\n--- Plotting Comprehensive Bar Charts by Model Type and Dataset ---")

    model_groups_to_plot = {
        "rl_tuned": comparison_df[comparison_df['model_group'] == 'RL-Tuned Models'],
        "distilled": comparison_df[comparison_df['model_group'] == 'Distilled Models']
    }

    for group_key, group_df in model_groups_to_plot.items():
        if group_df.empty:
            continue
        
        print(f"\nProcessing group: {group_key.replace('_', ' ').title()} Models...")
        datasets_in_group = group_df['data_name'].unique()

        for dataset in datasets_in_group:
            df_for_plot = group_df[group_df['data_name'] == dataset].copy()
            
            group_title_name = group_key.replace('_', ' ').title()
            plot_title = f"3-Way Comparison on {group_title_name} Models - Dataset: {dataset.upper()}"
            output_filename = f"figs/{group_key}_3way_comparison_{dataset}.png"

            plot_charts(
                df=df_for_plot,
                plot_title=plot_title,
                output_filename=output_filename
            )

if __name__ == "__main__":
    main()