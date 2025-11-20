"""
pip install -r requirements.txt
git clone https://github.com/openai/prm800k.git
cd prm800k && pip install -e . && cd ..
"""

import sys
import os
import sys
from pathlib import Path
from time import time
from dataclasses import dataclass
from typing import Literal
import random
import asyncio
import json
import re

def setup_prm800k():
    with open('/root/autodl-tmp/prm800k/prm800k/grading/grader.py', 'r') as file:
        content = file.read()
        
    # Make a small modification to handle relative imports
    modified_content = content.replace(
        'from grading import math_normalize',
        'from . import math_normalize')

    # Write back to the file
    with open('/root/autodl-tmp/prm800k/prm800k/grading/grader.py', 'w') as file:
        file.write(modified_content)
    
    return True

try:
    from prm800k.grading.grader import grade_answer
except ImportError as e:
    grade_answer = None

# Loading the MATH Dataset and some utility classes/methods

@dataclass
class MATHQuestion:
  problem: str
  answer: str
  solution: str
  subject: str
  level: int
  unique_id: str

  def get_prompt(self, instruction=None) -> str:
    if instruction is None:
      return f"{self.problem}\n\nPlease enclose your final answer in <answer></answer> tags."
    else:
      return f"{instruction}\n\n{self.problem}\n\nPlease enclose your final answer in <answer></answer> tags."

  @staticmethod
  def parse_response_for_answer(response: str) -> str:
    answer_tag = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_tag:
      answer = answer_tag.group(1).strip()
      answer = answer.replace('$', '').strip()
      return answer

    boxed_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    boxed_matches = re.findall(boxed_pattern, response)
    if boxed_matches:
        return boxed_matches[-1].strip()

    display_math = re.findall(r'\$\$\s*(.+?)\s*\$\$', response, re.DOTALL)
    if display_math:
        last_math = display_math[-1].strip()
        boxed_in_math = re.search(boxed_pattern, last_math)
        if boxed_in_math:
            return boxed_in_math.group(1).strip()

    return ""

  # def parse_response_for_answer(response: str) -> str:
  #   answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
  #   if answer_match is None:
  #       return ""
  #   return answer_match.group(1)


def load_questions(split: Literal["train", "test"], train_size: int = 1000, test_size: int = 100) -> list[MATHQuestion]:
  with open(f"prm800k/prm800k/math_splits/{split}.jsonl") as f:
    max_size = train_size if split == "train" else test_size
    raw_data = [json.loads(line) for line in f][:max_size]
  return [
     MATHQuestion(**d)for d in raw_data
  ]

# 使用时
train_dataset = load_questions("train", train_size=1000, test_size=500)  # train加载1000条
test_dataset = load_questions("test", train_size=1000, test_size=500)    # test加载500条

# We strongly recommend using the provided grading functions
def grade_question(question: MATHQuestion, model_answer: str) -> bool:
  return grade_answer(model_answer, question.answer)

def eval_model_answers(dataset: list[MATHQuestion], model_answers: list[str]) -> list[bool]:
  "Simple convenience function to evaluate a list of model answers  to a list of questions. Note that the answer must first be extracted from the response before being passed in here."
  return [grade_question(question, answer) for question, answer in zip(dataset, model_answers, strict=True)]


# if __name__ == "__main__":
#     setup_prm800k()

#     # Load the train and test datasets -- you should construct your few-shot prompts using only questions from the train dataset and then evaluate them on the test dataset
#     train_dataset = load_questions("train")
#     test_dataset = load_questions("test")
#     print(len(train_dataset), len(test_dataset))
