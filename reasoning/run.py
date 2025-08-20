import os
import json
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # model parameters
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-0.6B')

    parser.add_argument('--data_name', type=str, default='gpqa', 
                        help='Choose among: gpqa, gsm8k, mmlu, icraft, imedqa')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    
    parser.add_argument('--prompt_intervention', action='store_true',
                        help='Whether to use prompt intervention: `Answer only if you are confident. Otherwise, say "I am not sure."`')

    return parser

if __name__ == "__main__":

    args = get_args_parser().parse_args()
    model_name = args.model_name
    data_name = args.data_name

    if model_name == "microsoft/Phi-4-mini-flash-reasoning":
        os.environ["VLLM_ATTENTION_BACKEND"] = "DIFFERENTIAL_FLASH_ATTN"

    from inference import Inference
    from data import get_dataset_generator
    if 'mip' in data_name:
        results_dir = "results_mip"
    else:
        results_dir = "results"

    if args.prompt_intervention:
        additional_format_prompt = "Answer only if you are confident. Otherwise, say \"I am not sure.\""
        data_generator = get_dataset_generator(data_name, additional_format_prompt=additional_format_prompt)
        results_dir += "_prompt_intervention"
    else:
        data_generator = get_dataset_generator(data_name)

    begin_index = 0
    output_file = f"{results_dir}/{model_name}/{data_name}_results.jsonl"
    # ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        # get the index of the last line
        with open(output_file, "r") as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1]
                last_row = json.loads(last_line)
                begin_index = last_row.get("index", 0) + 1
    else:
        with open(output_file, "w") as f:
            f.write("")
    
    # if begin_index > 0:
    #     print(f"Some thread has already been processing the data. Exiting for now. ")
    #     exit(0)

    if begin_index >= len(data_generator.dataset):
        print(f"All data has been processed. No new data to process from index {begin_index}.")
        exit(0)

    inference_instance = Inference(model_name=model_name)
    data_generator.run_inference(inference_instance, output_file, begin_index, batch_size=args.batch_size)