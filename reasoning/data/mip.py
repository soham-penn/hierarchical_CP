from .base import BaseDatasetGenerator
import requests
import datasets
from datasets import load_dataset

class MIPGenerator(BaseDatasetGenerator):
    """
    This is not a multiple choice dataset.
    Answers are numeric
    """

    def __init__(
        self,
        dataset = "gsm8k", 
        **kwargs
    ):
        super().__init__(**kwargs)
        dataset_links = {
            "formula": ("https://raw.githubusercontent.com/tianyi-lab/MiP-Overthinking/refs/heads/main/data/formula.json", "_data/mip/formula"),
            "math500": ("https://raw.githubusercontent.com/tianyi-lab/MiP-Overthinking/refs/heads/main/data/math.json", "_data/mip/math500"),
            "gsm8k": ("https://raw.githubusercontent.com/tianyi-lab/MiP-Overthinking/refs/heads/main/data/gsm8k.json", "_data/mip/gsm8k"),
            "svamp": ("https://raw.githubusercontent.com/tianyi-lab/MiP-Overthinking/refs/heads/main/data/svamp.json", "_data/mip/svamp")
        }

        self.dataset_name = dataset
        url, data_dir = dataset_links.get(dataset, (None, None))
        try:
            self.original_dataset = datasets.Dataset.load_from_disk(data_dir)
        except:
            response = requests.get(url)
            data = response.json()

            if dataset == "gsm8k":
                print(f"Before filtering, dataset size: {len(data)}")
                other_dataset = load_dataset("openai/gsm8k", "main", split="test")
                # remove instances already present in the other dataset
                other_questions = set(other_dataset[i]["question"] for i in range(200))
                data = [item for item in data if item["question"] not in other_questions]
                print(f"After filtering, dataset size: {len(data)}")

            self.original_dataset = datasets.Dataset.from_list(data)

        self.dataset = self.create_dataset()
        self.format_prompt = 'Please reason step by step, and put your final answer within \\boxed{}, e.g., Answer: \\boxed{45}'

    def remove_context(self, question: str) -> str:
        # remove the first half of all sentences in the question
        sentences = question.split('. ')
        question_without_context = '. '.join(sentences[len(sentences) // 2:]).strip()
        return question_without_context
    
    def create_dataset(self):
        dataset = []
        for q in self.original_dataset:
            q["question_without_context"] = q["insufficient_question"]
            del q["insufficient_question"]
            if 'answer' in q:
                q["ref_answer"] = q["answer"]
                del q["answer"]
            dataset.append(q)

        print(f"Loaded {len(dataset)} examples from MIP-{self.dataset_name} dataset.")
        return dataset

if __name__ == "__main__":
    # Example usage
    mip = MIPGenerator()
    print(f"MIP dataset size: {len(mip.dataset)}")

    for i in range(3):
        print(mip.dataset[i])
    