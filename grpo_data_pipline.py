from collections import Counter, defaultdict
import pandas as pd
from typing import List
from datasets import Dataset
from math_datasets import setup_prm800k, load_questions, MATHQuestion
import re

# Helper functions to extract answers from different formats
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# Function to prepare the dataset for GRPO
def get_questions(questions: List[MATHQuestion]) -> Dataset:
    # Convert List[MATHQuestion] to dicts for Dataset creation
    data_dicts = [
        {
            "question": q.problem,
            "answer": q.answer,
        }
        for q in questions
    ]
    
    dataset = Dataset.from_list(data_dicts)
    
    dataset = dataset.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": x["answer"],
        }
    )
    return dataset

def analyze_dataset(dataset: List[MATHQuestion]):
  levels = [q.level for q in dataset]
  subjects = [q.subject for q in dataset]
  level_counts = Counter(levels)
  subject_counts = Counter(subjects)
  print("Level Distribution:")
  print("-" * 40)

  for level in sorted(level_counts.keys()):
    count = level_counts[level]
    percentage = (count / len(dataset)) * 100
    print(f"Level {level}: {count:4d} samples ({percentage:5.2f}%)")

  print("\nSubject Distribution:")
  print("-" * 40)
  for subject in sorted(subject_counts.keys()):
    count = subject_counts[subject]
    percentage = (count / len(dataset)) * 100
    print(f"  {subject:25s}: {count:4d} samples ({percentage:5.2f}%)")

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def convert_math_to_sft_format(train_dataset): 
    return Dataset.from_list([{"text": f"{q.get_prompt()}\n\n{q.solution}\n\n<answer>{q.answer}</answer>"} for q in train_dataset])

if __name__ == "__main__":
    setup_prm800k()

    # Load the train and test datasets -- you should construct your few-shot prompts using only questions from the train dataset and then evaluate them on the test dataset
    train_dataset = load_questions("train")
    test_dataset = load_questions("test")
    grpo_train_dataset = get_questions(train_dataset)
    print(grpo_train_dataset)
    # print(len(train_dataset), len(test_dataset))
    # analyze_dataset(train_dataset)
    # analyze_dataset(test_dataset)
    # train_dataset_sft = convert_math_to_sft_format(train_dataset)
    # print(len(train_dataset_sft))