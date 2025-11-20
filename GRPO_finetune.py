"""
pip install unsloth 
pip install transformers==4.56.2
pip install --no-deps trl==0.22.2
"""

import os
# Set offline mode immediately to prevent any HF connection attempts
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import re
from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer
import numpy as np
from math_datasets import MATHQuestion, load_questions, eval_model_answers
from grpo_data_pipline import get_questions, extract_xml_answer
from trl import GRPOConfig, GRPOTrainer
from prm800k.prm800k.grading.grader import grade_answer

lora_rank = 16
max_prompt_length = 1024

# os.environ["HF_HUB_OFFLINE"] = "1" # Moved to top
max_seq_length=2048

def load_model(model_name, max_seq_length=2048):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        local_files_only=True,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        full_finetuning=False,
        gpu_memory_utilization=0.6,
    )
    
    return model, tokenizer


def generate_text(model, tokenizer, messages, max_new_tokens=256, temperature=0.7, top_p=0.8, top_k=20):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    output = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        streamer=streamer,
    )
    
    return output

from typing import TypedDict
from time import time

device = "cuda" if torch.cuda.is_available() else "cpu"

class Message(TypedDict):
    role: str
    content: str

def get_few_shot_prompt(prompts_and_responses: list[tuple[str, str]]) -> list[dict]:
  """
  Formats a set of few-shot examples into something ingestible by the anthropic api client.

  Args:
    prompts_and_responses: A list of paired prompts and responses -- the prompts and separators are assumed to not contain the human and assistant separators.
  """
  messages = []
  for p, r in prompts_and_responses:
    # assert HUMAN_PROMPT not in p, "No need to place the human separator in the prompts!"
    # assert AI_PROMPT not in r, "No need to place the assistant separator in the responses!"
    messages.append(
        {
            "role": "user",
            "content": p
        }
    )
    messages.append(
        {
            "role": "assistant",
            "content": r
        }
    )

  return messages

few_shot_prompt = get_few_shot_prompt([("What is 2 + 2?", "2 + 2 = 4."), ("What is 49*7?", "49 * 7 = 343.")])
print(f"Few Shot Prompt Messages:\n{few_shot_prompt}")

def construct_gold_few_shot_examples(
    train_dataset: list[MATHQuestion],
    num_examples: int = 5
) -> list[tuple[str, str]]:
    # Select a subset of training examples
    selected_examples = train_dataset[:num_examples]

    few_shot_examples = []
    for question in selected_examples:
        prompt = question.get_prompt()
        # Format the answer with the <answer></answer> tags as expected
        gold_answer = f"<answer>{question.answer}</answer>"
        # print(question.parse_response_for_answer(gold_answer))
        few_shot_examples.append((prompt, gold_answer))

    return few_shot_examples


def get_message_with_few_shot_prompt(
    few_shot_prompt: list[Message],
    prompt: str,
    model_name: str,
    model,
    tokenizer,
    device,
    max_new_tokens: int = 3072,
) -> str:
    messages = few_shot_prompt + [
        {
            "role": "user",
            "content": prompt
        }
    ]

    start = time()
    response = generate_response(messages, max_new_tokens, model, tokenizer, device)
    print(f"Got response from {model_name} after {time() - start:.2f}s")
    
    return response

def generate_response(
    messages: list[Message],
    max_new_tokens: int,
    model,
    tokenizer,
    device
) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    return response

def get_messages_with_few_shot_prompt(
    few_shot_prompt: list[Message],
    prompts: list[str],
    model_name: str,
    model,
    tokenizer,
    device,
) -> list[str]:
    messages = []
    for i, p in enumerate(prompts):
        print(f"Processing {i+1}/{len(prompts)}...")
        response = get_message_with_few_shot_prompt(
            few_shot_prompt,
            prompt=p,
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        messages.append(response)
    return messages


# Reward function that checks if the answer is correct
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(
        "-" * 20,
        f"Question:\n{q}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extracted_responses[0]}",
    )
    
    rewards = []
    for r, a in zip(extracted_responses, answer):
        # Use grade_answer for robust mathematical comparison
        # grade_answer(model_answer, ground_truth)
        try:
            is_correct = grade_answer(r, a)
        except Exception:
            is_correct = False
        rewards.append(2.0 if is_correct else 0.0)
            
    return rewards


# Reward function that checks if the answer is an integer
def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


# Reward function that checks if the completion follows the strict format
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


# Reward function that checks if the completion follows a more relaxed format
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


# Reward function that counts XML tags and penalizes extra content
def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def evaluate_model_performance(
    test_dataset: list[MATHQuestion],
    few_shot_examples: list[tuple[str, str]],
    model_name: str,
    model, 
    tokenizer, 
    device,
) -> tuple[str, float]:

    few_shot_prompt = get_few_shot_prompt(few_shot_examples)
    test_prompts = [q.get_prompt() for q in test_dataset]

    responses = get_messages_with_few_shot_prompt(
        few_shot_prompt=few_shot_prompt,
        prompts=test_prompts,
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    # print(responses)

    model_answers = [MATHQuestion.parse_response_for_answer(resp) for resp in responses]
    print(model_answers)

    results = eval_model_answers(test_dataset, model_answers)
    accuracy = np.mean(results)
    return model_name, accuracy



def main():
# 1920-1930
    os.environ["HF_HUB_OFFLINE"] = "1"
    train_dataset = load_questions("train")
    test_dataset = load_questions("test")
    
    # 使用本地模型路径或已经下载的模型
    model_name = "./Qwen3-1.7B" 
    
    try:
        model, tokenizer = load_model(model_name=model_name)
    except Exception as e:
        print(f"Error loading model from {model_name}: {e}")
        print("Please ensure the model is downloaded locally.")
        return

    num_few_shot_gold = 3
    gold_examples = construct_gold_few_shot_examples(train_dataset, num_few_shot_gold)

    model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)   

    train_dataset_grpo = get_questions(train_dataset)
    training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # Increase to 4 for smoother training
    num_generations=6,  # Decrease if out of memory
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps=250,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",  # Can use Weights & Biases
    output_dir="outputs",
)

    trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=train_dataset_grpo,
)


    trainer_stats = trainer.train()
    
    model.save_pretrained("qwen_weak_grpo_model")  # Local saving
    tokenizer.save_pretrained("qwen_weak_grpo_model")
    
    model_name, acc = evaluate_model_performance(
        test_dataset = test_dataset,
        few_shot_examples = gold_examples,
        model_name = "qwen_weak_grpo_model",
        model = model,
        tokenizer = tokenizer,
        device = "cuda")
    print(f"Model: {model_name}, Accuracy: {acc}")

if __name__ == "__main__":
    main()