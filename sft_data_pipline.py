from collections import Counter, defaultdict
import pandas as pd
from typing import List
from datasets import Dataset
from math_datasets import setup_prm800k, load_questions, MATHQuestion

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
      

def convert_math_to_sft_format(train_dataset): 
    return Dataset.from_list([{"text": f"{q.get_prompt()}\n\n{q.solution}\n\n<answer>{q.answer}</answer>"} for q in train_dataset])

if __name__ == "__main__":
    setup_prm800k()

    # Load the train and test datasets -- you should construct your few-shot prompts using only questions from the train dataset and then evaluate them on the test dataset
    train_dataset = load_questions("train")
    test_dataset = load_questions("test")
    print(len(train_dataset), len(test_dataset))
    analyze_dataset(train_dataset)
    analyze_dataset(test_dataset)
    train_dataset_sft = convert_math_to_sft_format(train_dataset)
    print(len(train_dataset_sft))