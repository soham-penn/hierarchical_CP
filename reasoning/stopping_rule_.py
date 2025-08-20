from stopping_rules import UncertaintyStoppingRule, tokenize_string
from transformers import AutoTokenizer

class UncertaintyArrivalStoppingRule(UncertaintyStoppingRule):
    """
    A stopping rule based on the uncertainty of the model's predictions.
    This is a placeholder for a more complex implementation.
    """
    def __init__(self, model_name, lazy_interval: int = 1, quantile: float = 0.95):
        super().__init__(model_name, lazy_interval, quantile)

    def train(self, tokenized_reasoning_traces):
        raise NotImplementedError("Subclasses should implement this method.")

    def _should_stop(self) -> bool:
        raise NotImplementedError("Subclasses should implement this method.")

    def uncertainty_count(self, text):
        """Count the occurrences of uncertainty phrases in the text."""
        count = 0
        # Ensure text is lowercase for matching
        words = text.lower()
        for phrase in self.uncertainty_phrases:
            #if words.count(phrase.lower()) != 0:
            #    print(f"Found uncertainty phrase: {phrase} in text: {text} and it appears {words.count(phrase.lower())} times.")
            count += words.count(phrase.lower())
        # @TODO: Implement a more sophisticated counting mechanism if needed.
        raise NotImplementedError("Subclasses should implement this method.")
    

if __name__ == "__main__":
    model_name = "Qwen/Qwen3-32B"
    stopping_rule = UncertaintyArrivalStoppingRule(model_name = model_name, quantile=0.95)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    reasoning_trace_sample = 'Wait, the problem does not provide the infrastructure needed to solve the task. I\'m not entirely sure what to do next.'

    # test whether uncertainty is calculated correctly
    tokenized_sample = tokenize_string(reasoning_trace_sample, tokenizer)
    uncertainty = stopping_rule.uncertainty_count(tokenized_sample)
    print(f"Uncertainty for sample: {uncertainty}")