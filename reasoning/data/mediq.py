import requests
import datasets
import jsonlines
from .base import BaseDatasetGenerator

class _MediQSubDatasetGenerator(BaseDatasetGenerator):
    """Private class for constructing the MediQ sub-benchmarks, iMedQA and iCRAFT-MD. For evaluation, you probably want `MediQDataset` instead."""

    def __init__(
        self,
        data_dir="_data/mediq/icraftmd",
        data_url="https://raw.githubusercontent.com/stellalisy/mediQ/refs/heads/main/data/all_craft_md.jsonl",
        exclude_sample_ids=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        try:
            self.original_dataset = datasets.Dataset.load_from_disk(data_dir)
        except:
            response = requests.get(data_url)
            response.raise_for_status()

            # Response is a jsonlines file, rather than a json file, so parse it appropriately
            lines = response.text.split("\n")
            lines = [line for line in lines if line]  # Filter out any blank lines
            data = jsonlines.Reader(lines)

            self.original_dataset = datasets.Dataset.from_list(list(data))

            self.original_dataset.save_to_disk(data_dir)

        if exclude_sample_ids is not None:
            self.original_dataset = self.original_dataset.filter(
                lambda x: x["id"] not in exclude_sample_ids
            )
        
        self.dataset = self.create_dataset()
        if len(self.dataset) > 200:
            self.dataset = self.dataset[:200]

    def create_dataset(self):
        dataset = []
        for item in self.original_dataset:
            # Add a '.' to the end of each context sentence if needed
            context = [(c + "." if not c.endswith(".") else c) for c in item["context"]]

            question = item["question"]

            try:
                choices_text, correct_answer_index = self.shuffle_choices(
                    list(item["options"].values()), item["answer"]
                )

                full_question = f"Context: {' '.join(context)}\nQuestion: {question}\nChoices:\n{choices_text}\nAnswer: "
                question_without_context = f"Context: {context[0]}\nQuestion: {question}\nChoices:\n{choices_text}\nAnswer: "

                dataset.append({
                    "question": full_question,
                    "question_without_context": question_without_context,
                    "answer": item["answer"],
                    "ref_answer": correct_answer_index,
                })
            except:
                pass
        return dataset

class iCRAFTGenerator(_MediQSubDatasetGenerator):
    """iCRAFT-MD sub-benchmark of the MediQ dataset."""

    def __init__(self, data_dir="_data/mediq/icraftmd", data_url="https://raw.githubusercontent.com/stellalisy/mediQ/refs/heads/main/data/all_craft_md.jsonl", **kwargs):
        super().__init__(data_dir, data_url, **kwargs)

class iMEDQAGenerator(_MediQSubDatasetGenerator):
    """iMED-QA sub-benchmark of the MediQ dataset."""

    def __init__(self, data_dir="_data/mediq/imedqa", data_url="https://raw.githubusercontent.com/stellalisy/mediQ/refs/heads/main/data/all_dev_good.jsonl", **kwargs):
        super().__init__(data_dir, data_url, exclude_sample_ids={224, 298, 779}, **kwargs)

if __name__ == "__main__":
    # Example usage
    icraft_gen = iCRAFTGenerator()
    print(f"iCRAFT-MD dataset size: {len(icraft_gen.dataset)}")

    for i in range(3):
        print(icraft_gen.dataset[i])

    imedqa_gen = iMEDQAGenerator()
    print(f"iMED-QA dataset size: {len(imedqa_gen.dataset)}")
    for i in range(3):
        print(imedqa_gen.dataset[i])
    
    