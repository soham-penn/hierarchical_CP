from datasets import load_dataset, concatenate_datasets, Features, Value
from .base import BaseDatasetGenerator

class AIMEGenerator(BaseDatasetGenerator):
    """
    This is not a multiple choice dataset.
    Answers are numeric
    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.original_dataset_1 = load_dataset("Maxwell-Jia/AIME_2024", "default", split='train')
        self.original_dataset_2 = load_dataset("math-ai/aime25", "default", split='test')

        # change the column names to be consistent
        # for dataset 1, change 'Problem' to 'question' and 'Answer' to 'answer'
        self.original_dataset_1 = self.original_dataset_1.rename_column("Problem", "question")
        self.original_dataset_1 = self.original_dataset_1.rename_column("Answer", "answer")
        # for dataset 2, change 'problem' to 'question'
        self.original_dataset_2 = self.original_dataset_2.rename_column("problem", "question")

        # now keep only the 'question' and 'answer' columns
        self.original_dataset_1 = self.original_dataset_1.remove_columns(
            [col for col in self.original_dataset_1.column_names if col not in ["question", "answer"]]
        )
        self.original_dataset_2 = self.original_dataset_2.remove_columns(
            [col for col in self.original_dataset_2.column_names if col not in ["question", "answer"]]
        )
        features = Features({"question": Value("string"), "answer": Value("string")})
        self.original_dataset_1 = self.original_dataset_1.cast(features)
        self.original_dataset_2 = self.original_dataset_2.cast(features)

        # concatenate the two datasets
        self.original_dataset = concatenate_datasets([self.original_dataset_1, self.original_dataset_2])

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
            sentences = q["question"].split('. ')
            if len(sentences) > 1:
                q["question_without_context"] = self.remove_context(q["question"])
                q["ref_answer"] = q["answer"]
                dataset.append(q)
        
        print(f"Loaded {len(dataset)} examples from AIME dataset.")
        return dataset
        

if __name__ == "__main__":
    aime_generator = AIMEGenerator()

    for i in range(3):
        print(aime_generator.dataset[i])
