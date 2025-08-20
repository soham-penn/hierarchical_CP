import os
import json
from collections import defaultdict
import pandas as pd

def collect_and_visualize_results(models, datasets, results_base_dir="results", alpha=0.1, report_filename="uncertainty_analysis_report.md"):
    """
    Collects results from analysis JSON files, processes them, and saves formatted tables to a report file.
    Now includes both category-grouped and dataset-grouped views with highlighting for rows with least checkmarks.

    Args:
        models (list): List of model names.
        datasets (list): List of dataset names.
        results_base_dir (str): Base directory where results are stored.
        alpha (float): Significance level for p-value (default is 0.05).
        report_filename (str): Name of the markdown report file to save.
    """

    all_results = defaultdict(lambda: defaultdict(dict))
    report_content = []

    # Iterate through models and datasets to find result files
    for model in models:
        for dataset in datasets:
            result_file_path = os.path.join(results_base_dir, model, f"{dataset}_uncertainty_results.jsonl")

            if os.path.exists(result_file_path):
                with open(result_file_path, 'r', encoding='utf-8') as f:
                    try:
                        results = json.load(f)
                        if "category_uncertainty_results" in results:
                            all_results[model][dataset] = results["category_uncertainty_results"]
                        else:
                            report_content.append(f"Warning: 'category_uncertainty_results' not found in {result_file_path}\n")
                    except json.JSONDecodeError:
                        report_content.append(f"Error decoding JSON from {result_file_path}\n")
            else:
                report_content.append(f"Result file not found: {result_file_path}\n")

    # Prepare data for category-grouped tables
    category_tables_data = defaultdict(lambda: defaultdict(dict))
    
    # Prepare data for dataset-grouped tables
    dataset_tables_data = defaultdict(lambda: defaultdict(dict))

    for model, datasets_results in all_results.items():
        for dataset, categories_results in datasets_results.items():
            for category, stats in categories_results.items():
                t_stat = stats.get("t_stat")
                p_value = stats.get("p_value")

                if t_stat is not None:
                    if t_stat <= 0:
                        t_is_negative = "0️⃣"
                    else:
                        t_is_negative = ""
                else:
                    t_is_negative = ""
                
                p_is_significant = "✅" if p_value is not None and p_value < alpha else ""

                # Store data for category-grouped view (using p_value significance as main indicator)
                category_tables_data[category][(model, dataset)] = {
                    "t_neg": p_is_significant if p_is_significant else t_is_negative,
                    "p_sig": p_is_significant
                }
                
                # Store data for dataset-grouped view (using p_value significance as main indicator)  
                dataset_tables_data[dataset][(model, category)] = {
                    "t_neg": p_is_significant if p_is_significant else t_is_negative,
                    "p_sig": p_is_significant
                }

    def create_count_table(df, label_col, unique_models):
        """Create a count table showing checkmarks and zero emojis for each row."""
        count_data = []
        
        for _, row in df.iterrows():
            label = row[label_col]
            checkmark_count = 0
            zero_count = 0
            
            for model in unique_models:
                value = row[model]
                if value == "✅":
                    checkmark_count += 1
                elif value == "0️⃣":
                    zero_count += 1
            
            count_data.append({
                label_col: label,
                "✅ Count": checkmark_count,
                "0️⃣ Count": zero_count,
                "Total Symbols": checkmark_count + zero_count
            })
        
        # Create DataFrame and sort by checkmark count (descending), then by zero count (descending)
        count_df = pd.DataFrame(count_data)
        count_df = count_df.sort_values(["✅ Count", "0️⃣ Count", "Total Symbols"], ascending=[False, False, False])
        
        return count_df

    def extract_keywords_from_category(category):
        """Extract meaningful keywords from category names for family analysis."""
        import re
        
        # Convert to lowercase and split by common separators
        words = re.findall(r'\b\w+\b', category.lower())
        
        # Filter out common stop words and short words
        stop_words = {'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'cant', 'wont', 'dont', 'isnt', 'arent', 'wasnt', 'werent', 'hasnt', 'havent', 'hadnt', 'didnt', 'doesnt', 'wouldnt', 'couldnt', 'shouldnt', 'mightnt'}
        
        # Keep words that are 3+ characters and not stop words
        meaningful_words = [word for word in words if len(word) >= 3 and word not in stop_words]
        
        return meaningful_words

    def aggregate_and_analyze_families(dataset_tables_data, unique_models):
        """Aggregate count data across all datasets and analyze keyword families."""
        # Collect all category data across datasets
        all_category_data = defaultdict(lambda: {'checkmarks': 0, 'zeros': 0, 'occurrences': 0})
        
        # Also collect model-specific data
        model_category_data = defaultdict(lambda: defaultdict(lambda: {'checkmarks': 0, 'zeros': 0, 'occurrences': 0}))
        
        for dataset, dataset_data in dataset_tables_data.items():
            for (model, category), values in dataset_data.items():
                # Overall aggregation
                if values['t_neg'] == "✅":
                    all_category_data[category]['checkmarks'] += 1
                    model_category_data[model][category]['checkmarks'] += 1
                elif values['t_neg'] == "0️⃣":
                    all_category_data[category]['zeros'] += 1
                    model_category_data[model][category]['zeros'] += 1
                
                all_category_data[category]['occurrences'] += 1
                model_category_data[model][category]['occurrences'] += 1
        
        # Create aggregated category table
        category_summary = []
        for category, data in all_category_data.items():
            # Calculate both types of success rates
            checkmark_rate = (data['checkmarks'] / data['occurrences'] * 100) if data['occurrences'] > 0 else 0
            combined_rate = ((data['checkmarks'] + data['zeros']) / data['occurrences'] * 100) if data['occurrences'] > 0 else 0
            
            category_summary.append({
                'Category': category,
                'Total ✅': data['checkmarks'],
                'Total 0️⃣': data['zeros'],
                'Total Occurrences': data['occurrences'],
                '✅ Success Rate': f"{checkmark_rate:.1f}%",
                '✅+0️⃣ Success Rate': f"{combined_rate:.1f}%"
            })
        
        category_df = pd.DataFrame(category_summary)
        category_df = category_df.sort_values(['Total ✅', 'Total 0️⃣'], ascending=[False, False])
        
        # Analyze keyword families (overall)
        keyword_scores = defaultdict(lambda: {'checkmarks': 0, 'zeros': 0, 'categories': set(), 'total_occurrences': 0})
        
        # Also analyze keyword families by model
        model_keyword_scores = defaultdict(lambda: defaultdict(lambda: {'checkmarks': 0, 'zeros': 0, 'categories': set(), 'total_occurrences': 0}))
        
        for category, data in all_category_data.items():
            keywords = extract_keywords_from_category(category)
            for keyword in keywords:
                keyword_scores[keyword]['checkmarks'] += data['checkmarks']
                keyword_scores[keyword]['zeros'] += data['zeros']
                keyword_scores[keyword]['categories'].add(category)
                keyword_scores[keyword]['total_occurrences'] += data['occurrences']
        
        # Model-specific keyword analysis
        for model in unique_models:
            for category, data in model_category_data[model].items():
                keywords = extract_keywords_from_category(category)
                for keyword in keywords:
                    model_keyword_scores[model][keyword]['checkmarks'] += data['checkmarks']
                    model_keyword_scores[model][keyword]['zeros'] += data['zeros']
                    model_keyword_scores[model][keyword]['categories'].add(category)
                    model_keyword_scores[model][keyword]['total_occurrences'] += data['occurrences']
        
        # Create overall keyword family analysis
        keyword_analysis = []
        for keyword, data in keyword_scores.items():
            if len(data['categories']) >= 2:  # Only include keywords that appear in multiple categories
                checkmark_success_rate = (data['checkmarks'] / data['total_occurrences'] * 100) if data['total_occurrences'] > 0 else 0
                combined_success_rate = ((data['checkmarks'] + data['zeros']) / data['total_occurrences'] * 100) if data['total_occurrences'] > 0 else 0
                
                keyword_analysis.append({
                    'Keyword': keyword,
                    'Total ✅': data['checkmarks'],
                    'Total 0️⃣': data['zeros'],
                    'Categories Count': len(data['categories']),
                    'Total Occurrences': data['total_occurrences'],
                    '✅ Success Rate': f"{checkmark_success_rate:.1f}%",
                    '✅+0️⃣ Success Rate': f"{combined_success_rate:.1f}%",
                    'Example Categories': ', '.join(sorted(list(data['categories']))[:3])  # Show first 3 categories
                })
        
        keyword_df = pd.DataFrame(keyword_analysis)
        if not keyword_df.empty:
            keyword_df = keyword_df.sort_values(['Total ✅', 'Categories Count'], ascending=[False, False])
        
        # Create model-specific keyword analysis
        model_keyword_dfs = {}
        for model in unique_models:
            model_keyword_analysis = []
            for keyword, data in model_keyword_scores[model].items():
                if len(data['categories']) >= 2:  # Only include keywords that appear in multiple categories
                    checkmark_success_rate = (data['checkmarks'] / data['total_occurrences'] * 100) if data['total_occurrences'] > 0 else 0
                    combined_success_rate = ((data['checkmarks'] + data['zeros']) / data['total_occurrences'] * 100) if data['total_occurrences'] > 0 else 0
                    
                    model_keyword_analysis.append({
                        'Keyword': keyword,
                        'Total ✅': data['checkmarks'],
                        'Total 0️⃣': data['zeros'],
                        'Categories Count': len(data['categories']),
                        'Total Occurrences': data['total_occurrences'],
                        '✅ Success Rate': f"{checkmark_success_rate:.1f}%",
                        '✅+0️⃣ Success Rate': f"{combined_success_rate:.1f}%",
                        'Example Categories': ', '.join(sorted(list(data['categories']))[:3])
                    })
            
            if model_keyword_analysis:
                model_df = pd.DataFrame(model_keyword_analysis)
                model_df = model_df.sort_values(['Total ✅', 'Categories Count'], ascending=[False, False])
                model_keyword_dfs[model] = model_df
            else:
                model_keyword_dfs[model] = pd.DataFrame()
        
        return category_df, keyword_df, model_category_data, model_keyword_dfs

    # Generate report content
    report_content.append("# Uncertainty Analysis Report\n\n")
    report_content.append(f"**Significance Level (alpha):** {alpha}\n\n")
    report_content.append("**Legend:**\n")
    report_content.append("- ✅ = Significant difference (p-value < alpha)\n")
    report_content.append("- 0️⃣ = t-statistic ≤ 0\n\n")
    report_content.append("**Success Rate Definitions:**\n")
    report_content.append("- **✅ Success Rate:** Proportion of checkmarks only (significant results)\n")
    report_content.append("- **✅+0️⃣ Success Rate:** Proportion of checkmarks plus zeros (significant results + non-positive t-statistics)\n\n")

    # Section 1: Results grouped by category
    report_content.append("# Results Grouped by Category\n\n")
    
    for category, category_data in sorted(category_tables_data.items()):
        report_content.append(f"## Category: `{category}`\n")
        report_content.append("---\n\n")

        unique_models = sorted(list(set([k[0] for k in category_data.keys()])))
        
        df_rows = []
        for dataset in sorted(list(set([k[1] for k in category_data.keys()]))):
            row_data = {"Dataset": dataset}
            for model in unique_models:
                key = (model, dataset)
                row_data[f"{model}_t_neg"] = category_data.get(key, {}).get("t_neg", "")
            df_rows.append(row_data)

        all_columns = ["Dataset"]
        for model in unique_models:
            all_columns.append(f"{model}_t_neg")

        df = pd.DataFrame(df_rows, columns=all_columns)

        display_columns = {"Dataset": "Dataset"}
        for model in unique_models:
            display_columns[f"{model}_t_neg"] = f"{model}"

        df.rename(columns=display_columns, inplace=True)
        
        report_content.append(df.to_markdown(index=False) + "\n\n")

    # Section 2: Results grouped by dataset
    report_content.append("---\n\n")
    report_content.append("# Results Grouped by Dataset\n\n")
    
    for dataset, dataset_data in sorted(dataset_tables_data.items()):
        report_content.append(f"## Dataset: `{dataset}`\n")
        report_content.append("---\n\n")

        unique_models = sorted(list(set([k[0] for k in dataset_data.keys()])))
        
        df_rows = []
        for category in sorted(list(set([k[1] for k in dataset_data.keys()]))):
            row_data = {"Category": category}
            for model in unique_models:
                key = (model, category)
                row_data[f"{model}_t_neg"] = dataset_data.get(key, {}).get("t_neg", "")
            df_rows.append(row_data)

        all_columns = ["Category"]
        for model in unique_models:
            all_columns.append(f"{model}_t_neg")

        df = pd.DataFrame(df_rows, columns=all_columns)

        display_columns = {"Category": "Category"}
        for model in unique_models:
            display_columns[f"{model}_t_neg"] = f"{model}"

        df.rename(columns=display_columns, inplace=True)
        
        report_content.append(df.to_markdown(index=False) + "\n\n")
        
        # Add count table with dual success rates
        count_df = create_count_table(df, "Category", unique_models)
        
        # Calculate success rates for each category in this dataset
        enhanced_count_data = []
        for _, row in count_df.iterrows():
            checkmark_count = row["✅ Count"]
            zero_count = row["0️⃣ Count"]
            total_models = len(unique_models)
            
            checkmark_rate = (checkmark_count / total_models * 100) if total_models > 0 else 0
            combined_rate = ((checkmark_count + zero_count) / total_models * 100) if total_models > 0 else 0
            
            enhanced_count_data.append({
                "Category": row["Category"],
                "✅ Count": checkmark_count,
                "0️⃣ Count": zero_count,
                "Total Symbols": row["Total Symbols"],
                "✅ Success Rate": f"{checkmark_rate:.1f}%",
                "✅+0️⃣ Success Rate": f"{combined_rate:.1f}%"
            })
        
        enhanced_count_df = pd.DataFrame(enhanced_count_data)
        enhanced_count_df = enhanced_count_df.sort_values(["✅ Count", "0️⃣ Count"], ascending=[False, False])
        
        report_content.append("### Summary Counts with Success Rates:\n\n")
        report_content.append(enhanced_count_df.to_markdown(index=False) + "\n\n")

    # Generate aggregated analysis
    report_content.append("---\n\n")
    report_content.append("# Aggregated Analysis Across All Datasets\n\n")
    
    # Get unique models for aggregation
    all_unique_models = set()
    for dataset_data in dataset_tables_data.values():
        for (model, category) in dataset_data.keys():
            all_unique_models.add(model)
    all_unique_models = sorted(list(all_unique_models))
    
    category_summary_df, keyword_family_df, model_category_data, model_keyword_dfs = aggregate_and_analyze_families(dataset_tables_data, all_unique_models)
    
    report_content.append("## Overall Category Performance\n\n")
    report_content.append("This table shows the total performance of each category across all datasets and models:\n\n")
    report_content.append(category_summary_df.to_markdown(index=False) + "\n\n")
    
    if not keyword_family_df.empty:
        report_content.append("## Keyword Family Analysis\n\n")
        report_content.append("This analysis identifies keyword families that appear across multiple categories and their effectiveness:\n\n")
        report_content.append(keyword_family_df.to_markdown(index=False) + "\n\n")
        
        # Add insights section
        report_content.append("### Key Insights:\n\n")
        
        if len(keyword_family_df) > 0:
            # Most effective keywords (by checkmark success rate)
            top_keywords_checkmark = keyword_family_df.nlargest(3, 'Total ✅')
            report_content.append("**Most Effective Keyword Families (by ✅ count):**\n")
            for _, row in top_keywords_checkmark.iterrows():
                report_content.append(f"- **{row['Keyword']}**: {row['Total ✅']} successes across {row['Categories Count']} categories ({row['✅ Success Rate']} ✅-only rate, {row['✅+0️⃣ Success Rate']} combined rate)\n")
            report_content.append("\n")
            
            # Most effective keywords by combined success rate
            keyword_family_df['combined_numeric'] = keyword_family_df['✅+0️⃣ Success Rate'].str.rstrip('%').astype(float)
            top_keywords_combined = keyword_family_df.nlargest(3, 'combined_numeric')
            report_content.append("**Most Effective Keyword Families (by ✅+0️⃣ success rate):**\n")
            for _, row in top_keywords_combined.iterrows():
                report_content.append(f"- **{row['Keyword']}**: {row['Total ✅']} ✅ + {row['Total 0️⃣']} 0️⃣ across {row['Categories Count']} categories ({row['✅ Success Rate']} ✅-only rate, {row['✅+0️⃣ Success Rate']} combined rate)\n")
            report_content.append("\n")
            
            # Least effective keywords (bottom 3 by checkmark count)
            bottom_keywords = keyword_family_df.nsmallest(3, 'Total ✅')
            report_content.append("**Least Effective Keyword Families (by ✅ count):**\n")
            for _, row in bottom_keywords.iterrows():
                report_content.append(f"- **{row['Keyword']}**: {row['Total ✅']} successes across {row['Categories Count']} categories ({row['✅ Success Rate']} ✅-only rate, {row['✅+0️⃣ Success Rate']} combined rate)\n")
            report_content.append("\n")

    # Define the output path for the report file
    report_output_path = report_filename

    # Write all collected content to the report file
    with open(report_output_path, 'w', encoding='utf-8') as f:
        f.writelines(report_content)

    print(f"\nAnalysis report saved to: {report_output_path}")

# Alternative function for HTML output with CSS highlighting
def collect_and_visualize_results_html(models, datasets, results_base_dir="results", alpha=0.05, report_filename="uncertainty_analysis_report.html"):
    """
    Same as above but generates HTML output with CSS-based highlighting for better visual distinction.
    """
    # [Previous data collection code would be the same...]
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Uncertainty Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            th { background-color: #f2f2f2; }
            .highlight-row { background-color: #ffeeee; font-weight: bold; }
            .checkmark { color: #28a745; }
            .zero-emoji { color: #ffc107; }
            h1, h2 { color: #333; }
            .legend { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>Uncertainty Analysis Report</h1>
        <div class="legend">
            <strong>Legend:</strong><br>
            ✅ = Negative t-statistic or significant p-value<br>
            0️⃣ = Zero t-statistic<br>
            <span style="background-color: #ffeeee; padding: 2px 4px;">Highlighted rows</span> = Rows with fewest checkmarks (need attention)<br>
            <strong>Success Rates:</strong><br>
            ✅ Success Rate = Proportion of checkmarks only<br>
            ✅+0️⃣ Success Rate = Proportion of checkmarks plus zeros
        </div>
        <!-- Tables would be generated here with HTML table format -->
    </body>
    </html>
    """
    
    # Implementation would convert pandas DataFrames to HTML tables with highlighting
    pass

# Define your models and datasets here
models = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-32B", "Qwen/QwQ-32B", "microsoft/Phi-4-reasoning", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
          "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
datasets = ["gpqa", "gsm8k", "mmlu", "icraft", "imedqa"]

if __name__ == "__main__":
    collect_and_visualize_results(models, datasets)