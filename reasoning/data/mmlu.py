import re
from .base import BaseDatasetGenerator
from datasets import load_dataset, concatenate_datasets

class MMLUMathGenerator(BaseDatasetGenerator):
    SUBSETS = ["college_mathematics", "abstract_algebra", "high_school_mathematics"]

    def __init__(
        self,
        split="test",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.subsets = self.SUBSETS
        self.split = split
        self.original_dataset = self.load_datasets()
        self.dataset = self.create_dataset()

    def load_datasets(self):
        all_datasets = []
        for subset in self.subsets:
            dataset = load_dataset("cais/mmlu", subset, split=self.split)
            all_datasets.append(dataset)
        return concatenate_datasets(all_datasets)

    def create_dataset(self):
        dataset = []
        for q in self.original_dataset:
            if re.search(self.context_regex_pattern, q["question"]):
                choices = q["choices"]
                choices_text, correct_answer_index = self.shuffle_choices(
                    choices, q["choices"][q["answer"]]
                )
                question_without_context = self.remove_context(q["question"])
                dataset.append({
                        'question': q["question"] + "\n" + choices_text,
                        'question_without_context': question_without_context + "\n" + choices_text,
                        'answer': q["choices"][q["answer"]],
                        'ref_answer': correct_answer_index
                    }
                )
        return dataset
    
if __name__ == "__main__":
    mmlu_math = MMLUMathGenerator()
    # print length of dataset with and without context
    print(f"Dataset without context: {len(mmlu_math.dataset)}")

    for i in range(3):
        print(mmlu_math.dataset[i])
