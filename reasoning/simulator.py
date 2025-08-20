import os
os.environ['HF_HOME'] = './hf_home'
os.environ['TRANSFORMERS_CACHE'] = './hf_home/hub'
from transformers import AutoTokenizer

import argparse
from utils import load_data
import json

from stopping_rules import *
from data import get_dataset_generator
from eda import detect_abstention

# The goal of this script is to set up a simulation environment for reasoning traces
# and to implement conformal decision-making methods.
# - a **simulation environment** that mimics how decoding happens in real time
#         - scan through the reasoning traces
#           - how many tokens there have been so far
#           - calculate the density of uncertainty so far
#           - stop generation when a decision is made
#       - create an abstract class and define the methods that need to be implemented for comformal decisions
#       - given prompt $x$ and completion $y$, produce transformation $h(y)$ which matches longer messages to shorter messages.
#         - in abstention settings, can be truncating reasoning traces, or setting to empty strings

def get_args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--model_name', type=str, default='microsoft/Phi-4-reasoning')
    parser.add_argument('--data_name', type=str, default='gsm8k',
                        help='Choose among: gpqa, gsm8k, mmlu, icraft, imedqa')
    parser.add_argument('--results_path', type=str, default='results',
                        help='Path to save the results')
    parser.add_argument('--stopping_rule', type=str, default='uncertainty',
                        help='Stopping rule to use for decoding. Choose among: uncertainty or length')
    parser.add_argument('--quantile', type=float, default=0.95,
                        help='Quantile to use for uncertainty stopping rule')
    return parser


class DecodingSimulator:
    """
    A class to simulate the decoding process of a reasoning trace.
    It generates tokens one by one and calculates the savings based on the stopping rule.
    """

    def __init__(self, tokenizer: AutoTokenizer, reasoning_trace: str):
        self.reasoning_trace = reasoning_trace
        self.encoded_trace = tokenize_string(reasoning_trace, tokenizer)
        self.current_position = 0
    
    def next_token(self, n: int = 1):
        """Simulate the next token generation based on the current position in the reasoning trace."""
        if self.current_position + n < len(self.encoded_trace):
            tokens = self.encoded_trace[self.current_position:self.current_position + n]
            self.current_position += n
            return tokens
        else:
            self.current_position = len(self.encoded_trace)  # Move to the end of the trace
            return None
        
    def calculate_savings(self):
        """
        Calculate the savings based on the current position in the reasoning trace.
        Return two values:
        - The number of tokens saved.
        - The percentage of tokens saved compared to the total length of the reasoning trace.
        """
        number_of_tokens = len(self.encoded_trace)
        tokens_saved = number_of_tokens - self.current_position 
        percentage_saved = (tokens_saved / number_of_tokens) * 100 if number_of_tokens > 0 else 0
        return tokens_saved, percentage_saved



class DecodingTestEnvironment:
    """
    A class to set up the decoding test environment.
    It initializes the stopping rule and runs the simulation for each reasoning trace in the dataset.
    """

    def __init__(self, model_name: str, data_name: str, stopping_rule: StoppingRule, mode: str = 'with_context', results_path: str = 'results',
                 split: str = 'test'):
        assert mode in ['with_context', 'without_context'], "Mode must be either 'with_context' or 'without_context'."
        self.model_name = model_name
        self.stopping_rule = stopping_rule
        self.interval = stopping_rule.lazy_interval
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        data = self.load_data(model_name, data_name, split=split, results_path=results_path)
        if split != 'full':
            self.data_generator = get_dataset_generator(data_name)
        self.mode = mode
        if data is None:
            raise FileNotFoundError(f"Data for {model_name} not found.")
        if self.mode == 'with_context':
            self.reasoning_traces = [row.get("thinking_content", "") for row in data]
            self.contents = [row.get("content", "") for row in data]
            self.reference_answers = [row.get("ref_answer", "") for row in data]
        else:
            self.reasoning_traces = [row.get("thinking_content_without_context", "") for row in data]
            self.contents = [row.get("content_without_context", "") for row in data]
            self.reference_answers = [row.get("ref_answer_without_context", "") for row in data]

    def load_data(self, model_name, data_name, split='test', results_path='results'):
        data = load_data(model_name, data_name, split=split, results_path=results_path)
        return data

    def run_simulation(self):
        tokens_saved_list = []
        percentage_saved_list = []
        early_stopping_count = 0
        accuracy_count = 0
        abstention_count = 0
        for idx, reasoning_trace in enumerate(self.reasoning_traces):
            self.simulator = DecodingSimulator(
                tokenizer=self.tokenizer,
                reasoning_trace=reasoning_trace
            )
            # print(f"Starting simulation for reasoning trace: {reasoning_trace}")
            
            while True:
                token = self.simulator.next_token(n=self.interval)
                if token is None:
                    break
                if self.stopping_rule.should_stop(token):
                    early_stopping_count += 1
                    if self.mode == 'with_context':
                        accuracy_count += self.data_generator.evaluate(self.contents[idx], self.reference_answers[idx])
                    else:
                        abstention_count += 1 - detect_abstention(self.contents[idx])
                    break
            tokens_saved, percentage_saved = self.simulator.calculate_savings()
            self.stopping_rule.clear()  # Clear the stopping rule for the next trace
            # print(f"Tokens saved: {tokens_saved}, Percentage saved: {percentage_saved:.2f}%")
            tokens_saved_list.append(tokens_saved)
            percentage_saved_list.append(percentage_saved)

        avg_tokens_saved = sum(tokens_saved_list) / len(tokens_saved_list) if tokens_saved_list else 0
        avg_percentage_saved = sum(percentage_saved_list) / len(percentage_saved_list) if percentage_saved_list else 0
        early_stopping_rate = early_stopping_count / len(self.reasoning_traces) * 100 if self.reasoning_traces else 0
        if self.mode == 'with_context':
            accuracy_dropped = accuracy_count / len(self.reasoning_traces) * 100 if self.reasoning_traces else 0
            return avg_tokens_saved, avg_percentage_saved, early_stopping_rate, accuracy_dropped
        else:
            abstention_increased = abstention_count / len(self.reasoning_traces) * 100 if self.reasoning_traces else 0
            return avg_tokens_saved, avg_percentage_saved, early_stopping_rate, abstention_increased


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    model_name = args.model_name
    data_name = args.data_name
    results_path = args.results_path
    split = "test"

    path = f"{results_path}/{model_name}/{data_name}_stopping_rule_results.jsonl"


    if args.stopping_rule == 'uncertainty':
        stopping_rule = UncertaintyStoppingRule(model_name, lazy_interval=100, quantile=args.quantile)
    else:
        stopping_rule = LengthStoppingRule(lazy_interval=100, quantile=args.quantile)

    if "mip" in results_path or data_name == 'aime':
        train_stopping_rule(stopping_rule, model_name, "gsm8k", results_path="results")
        split = "full"
    else:
        train_stopping_rule(stopping_rule, model_name, data_name, results_path=results_path)
        decoding_env = DecodingTestEnvironment(model_name=model_name, data_name=data_name, stopping_rule=stopping_rule, mode = 'with_context', results_path=results_path)
        avg_tokens_saved, avg_percentage_saved, early_stopping_rate, accuracy_dropped = decoding_env.run_simulation()

     
    decoding_env_without_context = DecodingTestEnvironment(model_name=model_name, data_name=data_name, stopping_rule=stopping_rule, mode = 'without_context', results_path=results_path, split=split)
    avg_tokens_saved_without_context, avg_percentage_saved_without_context, early_stopping_rate_without_context, abstention_increased = decoding_env_without_context.run_simulation()
    

    if "mip" in results_path or data_name == 'aime':
        results = {
            "stopping_rule": stopping_rule.__class__.__name__,
            "quantile": stopping_rule.quantile,
            "avg_tokens_saved_without_context": avg_tokens_saved_without_context,
            "avg_percentage_saved_without_context": avg_percentage_saved_without_context,
            "early_stopping_rate_without_context": early_stopping_rate_without_context,
            "abstention_increased_without_context": abstention_increased,
        }
    else:
        results = {
            "stopping_rule": stopping_rule.__class__.__name__,
            "quantile": stopping_rule.quantile,
            "avg_tokens_saved_with_context": avg_tokens_saved,
            "avg_percentage_saved_with_context": avg_percentage_saved,
            "early_stopping_rate_with_context": early_stopping_rate,
            "avg_tokens_saved_without_context": avg_tokens_saved_without_context,
            "avg_percentage_saved_without_context": avg_percentage_saved_without_context,
            "early_stopping_rate_without_context": early_stopping_rate_without_context,
            "accuracy_dropped_with_context": accuracy_dropped,
            "abstention_increased_without_context": abstention_increased,
        }

    # save the trained

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as file:
        file.write(json.dumps(results) + '\n')