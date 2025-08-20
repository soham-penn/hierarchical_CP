from utils import load_data
import os
os.environ['HF_HOME'] = './hf_home'
os.environ['TRANSFORMERS_CACHE'] = './hf_home/hub'
from typing import List
import numpy as np
from transformers import AutoTokenizer

def tokenize_string(string: str, tokenizer: AutoTokenizer) -> List[int]:
    """
    Tokenize the given string using the specified model's tokenizer.
    :param string: The string to tokenize.
    :param model_name: The name of the model to use for tokenization.
    :return: A list of token IDs.
    """
    encoded = tokenizer(string, return_tensors='pt')
    return encoded['input_ids'][0].tolist()

class StoppingRule:
    """
    Abstract class for stopping rules.
    The stopping rule monitors the decoded tokens so far and decides when to stop decoding.
    """
    def __init__(self, lazy_interval: int = 100, **kwargs):
        """
        Initialize the stopping rule with a lazy interval.
        :param lazy_interval: The number of tokens to skip before checking the stopping condition.
        """
        self.tokens_so_far = []
        self.lazy_interval = lazy_interval

    def lazy_mode(self):
        if len(self.tokens_so_far) % self.lazy_interval == 0:
            return False
        return True
    
    def clear(self):
        """
        Clear the tokens decoded so far.
        """
        self.tokens_so_far = []

    def should_stop(self, input: List) -> bool:
        self.tokens_so_far += input
        if self.lazy_mode():
            return False
        else:
            return self._should_stop()

    def _should_stop(self) -> bool:
        """
        This method should be implemented by subclasses to define the specific stopping criteria.
        :return: True if decoding should stop, False otherwise.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def train(self, tokenized_reasoning_traces: List[str]):
        """
        Train the stopping rule based on the provided tokenized reasoning traces.
        :param tokenized_reasoning_traces: A list of tokenized reasoning traces to train the stopping rule.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def load(self, path: str):
        """
        Load the stopping rule parameters from a file.
        :param path: The path to the file containing the stopping rule parameters.
        """
        with open(path, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                setattr(self, key, value)
    
    def save(self, path: str):
        """
        Save the stopping rule parameters to a file.
        :param path: The path to the file where the stopping rule parameters will be saved.
        """
        with open(path, 'w') as f:
            for key, value in self.__dict__.items():
                f.write(f"{key}: {value}\n")

class LengthStoppingRule(StoppingRule):
    """
    A simple stopping rule that stops decoding after a certain number of tokens.
    """
    def __init__(self, lazy_interval: int = 1, quantile: float = 0.95):
        super().__init__(lazy_interval)
        self.quantile = quantile
        self.length_threshold = None

    def train(self, tokenized_reasoning_traces):
        lengths = [len(trace) for trace in tokenized_reasoning_traces]
        n = len(lengths)
        self.length_threshold = int(np.quantile(lengths, self.quantile * (n+1) / n))

    def _should_stop(self) -> bool:
        return len(self.tokens_so_far) >= self.length_threshold

class UncertaintyStoppingRule(StoppingRule):
    """
    A stopping rule based on the uncertainty of the model's predictions.
    This is a placeholder for a more complex implementation.
    """
    def __init__(self, model_name, lazy_interval: int = 1, quantile: float = 0.95):
        super().__init__(lazy_interval)
        from detect.uncertainty_keywords import uncertainty_keywords_dict
        self.uncertainty_threshold = None
        self.uncertainty_phrases = uncertainty_keywords_dict['missing_information'] + \
            uncertainty_keywords_dict['doubt_speculation'] + \
            uncertainty_keywords_dict['questioning_the_premise']
                
        self.quantile = quantile
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.uncertainty_patience = 0
    
    def lazy_mode(self):
        if len(self.tokens_so_far) < 1000:
            return True
        return super().lazy_mode()

    def uncertainty_density(self, text):
        """Count the occurrences of uncertainty phrases in the text."""
        count = 0
        # Ensure text is lowercase for matching
        words = text.lower()
        for phrase in self.uncertainty_phrases:
            #if words.count(phrase.lower()) != 0:
            #    print(f"Found uncertainty phrase: {phrase} in text: {text} and it appears {words.count(phrase.lower())} times.")
            count += words.count(phrase.lower())
        return count/len(words) if len(words) > 0 else 0

    def calculate_uncertainty(self, tokenized_reasoning_trace):
        """
        Calculate the uncertainty of the model's predictions based on the tokenized reasoning trace.
        This is a placeholder for a more complex implementation.
        :param tokenized_reasoning_trace: The tokenized reasoning trace to calculate uncertainty for.
        :return: A float representing the uncertainty.
        """
        # Placeholder implementation, replace with actual uncertainty calculation logic
        # decode the tokens back to text
        
        text = self.tokenizer.decode(tokenized_reasoning_trace, skip_special_tokens=True)
        return self.uncertainty_density(text)

    def train(self, tokenized_reasoning_traces):
        uncertainties = [self.calculate_uncertainty(trace) for trace in tokenized_reasoning_traces]
        n = len(uncertainties)
        self.uncertainty_threshold = np.quantile(uncertainties, self.quantile * (n+1) / n)

    def _should_stop(self) -> bool:
        current_uncertainty = self.calculate_uncertainty(self.tokens_so_far)
        if current_uncertainty > self.uncertainty_threshold:
            if self.uncertainty_patience < 5:
                self.uncertainty_patience += 1
            else:
                self.uncertainty_patience = 0
                return True
        return False

def train_stopping_rule(stopping_rule: StoppingRule, model_name, data_name: str, results_path: str):
    """
    Train the stopping rule with the provided reasoning traces.
    :param stopping_rule: The stopping rule to train.
    :param reasoning_traces: A list of reasoning traces to use for training.
    """
    data = load_data(model_name, data_name, split='train', results_path=results_path)
    if data is None:
        raise FileNotFoundError(f"Data for {model_name} not found.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_reasoning_traces = [tokenize_string(row.get("thinking_content", ""), tokenizer) for row in data]
    stopping_rule.train(tokenized_reasoning_traces)



if __name__ == "__main__":
    model_name = "Qwen/Qwen3-32B"
    stopping_rule = UncertaintyStoppingRule(model_name = model_name, quantile=0.95)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    reasoning_trace_sample = 'Wait, the problem does not provide the infrastructure needed to solve the task. I\'m not entirely sure what to do next.'

    # test whether uncertainty is calculated correctly
    tokenized_sample = tokenize_string(reasoning_trace_sample, tokenizer)
    uncertainty = stopping_rule.calculate_uncertainty(tokenized_sample)
    print(f"Uncertainty for sample: {uncertainty}")