import re

from datasets import load_dataset
from .base import BaseDatasetGenerator

class GSM8KGenerator(BaseDatasetGenerator):
    """
    Filters GSM8K questions that contain
    [context]. [question] ?

    via regex

    then offers two versions of each question
    with and without context

    This is not a multiple choice dataset.
    Answers are numeric
    """

    def __init__(
        self,
        split="test",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.split = split
        self.original_dataset = load_dataset("openai/gsm8k", "main", split=split)
        self.dataset = self.create_dataset()[:200]
        self.format_prompt = 'Please reason step by step, and put your final answer within \\boxed{}, e.g., Answer: \\boxed{45}'

    def create_dataset(self):
        dataset = []
        for q in self.original_dataset:
            if re.search(self.context_regex_pattern, q["question"]):
                q["question_without_context"] = self.remove_context(q["question"])
                # answer choice is behind '#### '
                q["ref_answer"] = q["answer"].split("#### ")[-1].strip()
                dataset.append(q)
        return dataset
        

if __name__ == "__main__":
    gsm8k_generator = GSM8KGenerator()

    for i in range(3):
        print(gsm8k_generator.dataset[i])
    