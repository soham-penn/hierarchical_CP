# load data from https://github.com/tianyi-lab/MiP-Overthinking/
import json
import os
import requests


datasets = [
    ("Qwen/QwQ-32B", "math500", "https://raw.githubusercontent.com/tianyi-lab/MiP-Overthinking/refs/heads/main/response/math/MiP/QwQ/response.json"),
    ("Qwen/QwQ-32B", "gsm8k", "https://raw.githubusercontent.com/tianyi-lab/MiP-Overthinking/refs/heads/main/response/gsm8k/MiP/QwQ/response.json"),
    ("Qwen/QwQ-32B", "formula", "https://raw.githubusercontent.com/tianyi-lab/MiP-Overthinking/refs/heads/main/response/formula/MiP/QwQ/response.json"),
    ("Qwen/QwQ-32B", "svamp", "https://raw.githubusercontent.com/tianyi-lab/MiP-Overthinking/refs/heads/main/response/svamp/MiP/QwQ/response.json"),

    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "math500", "https://raw.githubusercontent.com/tianyi-lab/MiP-Overthinking/refs/heads/main/response/math/MiP/DSQ/response.json"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "gsm8k", "https://raw.githubusercontent.com/tianyi-lab/MiP-Overthinking/refs/heads/main/response/gsm8k/MiP/DSQ/response.json"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "formula", "https://raw.githubusercontent.com/tianyi-lab/MiP-Overthinking/refs/heads/main/response/formula/MiP/DSQ/response.json"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "svamp", "https://raw.githubusercontent.com/tianyi-lab/MiP-Overthinking/refs/heads/main/response/svamp/MiP/DSQ/response.json")
]

def download_and_save_dataset(model_name, dataset_name, url):
    # Download and parse the JSON
    response = requests.get(url)
    data = response.json()

    output_dir = f"{model_name}/"
    # create directory
    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))

    output_file = os.path.join(output_dir, f"{dataset_name}_results.jsonl")

    # for each row, change the following column name: insufficient_question -> question_without_context;
    for row in data:
        if "insufficient_question" in row:
            row["question_without_context"] = row.pop("insufficient_question")
        
        generated_text = row.pop("model_answer")
        if "</think>" in generated_text:
            content = generated_text.split("</think>")[1].strip()
            thinking_content = generated_text.split("</think>")[0].strip()
        else:
            # If the </think> token is not found, it assumes that the entire output is thinking_content.
            thinking_content = generated_text.strip()
            content = ""
        row["thinking_content_without_context"] = thinking_content
        row["content_without_context"] = content

        # only keep relevant columns
        relevant_columns = [
            "question_without_context",
            "thinking_content_without_context",
            "content_without_context"
        ]
        row = {key: row[key] for key in relevant_columns if key in row}

        with open(output_file, "a") as f:
            f.write(json.dumps(row) + "\n")


for dataset in datasets:
    model_name, dataset_name, url = dataset
    download_and_save_dataset(model_name, dataset_name, url)