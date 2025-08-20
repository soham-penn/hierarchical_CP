import re
from .base import BaseDatasetGenerator

from datasets import load_dataset

class GPQAGenerator(BaseDatasetGenerator):
    """
    Multiple choice graduate level science questions
    that are not googleable.

    diamond is what DeepSeek evaluates
    there's only one split called train
    original dataset has a total of 198 questions

    after filtering questions that contain a clear context
    total dataset size with/without context is 132
    """

    def __init__(self, subset="gpqa_diamond", **kwargs):
        super().__init__(**kwargs)
        self.subset = subset
        try:
            self.original_dataset = load_dataset("Idavidrein/gpqa", subset, split="train")
        except FileNotFoundError:
            print("Falling back to local CSV")
            self.original_dataset = load_dataset("csv", data_files="_data/gpqa/gpqa_diamond.csv")
            
        self.dataset = self.create_dataset()

    def create_dataset(self):
        dataset = []
        for q in self.original_dataset:
            if re.search(self.context_regex_pattern, q["Question"]):
                question = q["Question"]
                question_without_context = self.remove_context(question)
                choices = [
                    self._preprocess(q["Incorrect Answer 1"]),
                    self._preprocess(q["Incorrect Answer 2"]),
                    self._preprocess(q["Incorrect Answer 3"]),
                    self._preprocess(q["Correct Answer"]),
                ]
                choices_text, correct_answer_index = self.shuffle_choices(
                    choices, self._preprocess(q["Correct Answer"])
                )
                dataset.append({
                        'question': question + "\n" + choices_text,
                        'question_without_context': question_without_context + "\n" + choices_text,
                        'answer': q["Correct Answer"],
                        'ref_answer': correct_answer_index
                    }
                )
        return dataset

    def _preprocess(self, text):
        if text is None:
            return " "
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

if __name__ == "__main__":
    gpqa = GPQAGenerator()
    # print length of dataset with and without context
    print(f"Dataset with context: {len(gpqa.dataset)}")

    for i in range(3):
        print(gpqa.dataset[i])
    