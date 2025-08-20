import os
import json
import argparse # Added for argparse in the standalone script

def aggregate_and_deduplicate_features(results_base_dir="results"):
    """
    Aggregates top feature names from JSON analysis results across models and datasets,
    excluding those where classification_accuracy is <= 0.5,
    and returns a deduplicated list.

    Args:
        results_base_dir (str): The base directory where your model results are stored.

    Returns:
        list: A sorted list of unique top feature names found across qualified analyses.
    """
    all_features = []
    
    # Walk through the results directory to find all JSON files
    for root, dirs, files in os.walk(results_base_dir):
        for file in files:
            # Check for both possible naming conventions for classification results
            if file.endswith("_classification_results.jsonl") or file.endswith("_top_features_only.jsonl"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # --- NEW CHECK HERE ---
                        # Check classification_accuracy
                        accuracy = data.get("classification_accuracy")
                        
                        # Handle cases where accuracy might be 'N/A' or a string, or missing
                        if isinstance(accuracy, (int, float)):
                            if accuracy <= 0.5:
                                print(f"  Skipping {file_path}: Classification accuracy ({accuracy:.2f}) is <= 0.5.")
                                continue # Skip this file
                        else:
                            # If accuracy is not a number (e.g., 'N/A' or missing), assume it's not good enough
                            print(f"  Skipping {file_path}: Classification accuracy is not a valid number or missing.")
                            continue # Skip this file

                        # Proceed to extract features if accuracy is > 0.5
                        if 'top_feature_importances' in data and isinstance(data['top_feature_importances'], list):
                            for item in data['top_feature_importances']:
                                if isinstance(item, dict) and 'feature' in item:
                                    all_features.append(item['feature'])
                                elif isinstance(item, str):
                                    all_features.append(item)
                                    
                        elif 'top_features' in data and isinstance(data['top_features'], list):
                            for item in data['top_features']:
                                if isinstance(item, dict) and 'feature' in item:
                                    all_features.append(item['feature'])
                                elif isinstance(item, str):
                                    all_features.append(item)
                                    
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {file_path}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred while processing {file_path}: {e}")

    # Remove duplicates by converting to a set and then back to a list
    unique_features = sorted(list(set(all_features)))
    
    return unique_features

# --- main function context (as provided previously) ---
def main():
    """Main function to run the feature aggregation."""
    parser = argparse.ArgumentParser('Feature Aggregation', add_help=False)
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Base directory where all model/dataset results are stored.')
    args = parser.parse_args()

    print(f"Aggregating features from: {args.results_dir}")
    aggregated_unique_features = aggregate_and_deduplicate_features(args.results_dir)

    output_path = os.path.join(args.results_dir, "aggregated_unique_features_filtered.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(aggregated_unique_features, f, indent=4)

    print(f"\n--- Aggregation Complete ---")
    print(f"Total unique features found: {len(aggregated_unique_features)}")
    print(f"Aggregated unique features saved to: {output_path}")
    print("Example top 10 features:", aggregated_unique_features[:10])


if __name__ == "__main__":
    main()