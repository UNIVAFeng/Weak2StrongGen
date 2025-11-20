"""
pip install unsloth 
pip install transformers==4.56.2
pip install --no-deps trl==0.22.2
"""

import os
import re
from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer
import numpy as np
from math_datasets import MATHQuestion, load_questions, eval_model_answers

os.environ["HF_HUB_OFFLINE"] = "1"

def load_model(model_name, max_seq_length=2048, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        local_files_only=True,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
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
    max_new_tokens: int = 512,
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

    model_answers = [MATHQuestion.parse_response_for_answer(resp) for resp in responses]

    results = eval_model_answers(test_dataset, model_answers)
    accuracy = np.mean(results)
    return model_name, accuracy



def main():
    train_dataset = load_questions("train")
    test_dataset = load_questions("test")
    qwen_model, qwen_tokenizer = load_model(model_name="./Qwen3-0.6B")
    # messages = [
    #     {"role": "user", "content": "你好，请介绍一下自己。"}
    # ]
    
    # # 生成文本
    # print("\n正在生成文本...")
    # generate_text(model, tokenizer, messages)
    num_few_shot_gold = 3
    gold_examples = construct_gold_few_shot_examples(train_dataset, num_few_shot_gold)
    

    finetned_qwen_model = FastLanguageModel.get_peft_model(
    qwen_model,
    r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,  # Best to choose alpha = rank or rank*2
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,   # We support rank stabilized LoRA
    loftq_config = None,  # And LoftQ
)    

    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset_sft,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use TrackIO/WandB etc
    ),
)

    model_name, acc = evaluate_model_performance(
        test_dataset = test_dataset,
        few_shot_examples = gold_examples,
        model_name = "qwen3-0.6B",
        model = qwen_model,
        tokenizer = qwen_tokenizer,
        device = "cuda")
    print(f"Model: {model_name}, Accuracy: {acc}")
if __name__ == "__main__":
    main()