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
from sft_data_pipline import convert_math_to_sft_format
from trl import SFTTrainer, SFTConfig

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



def finetune_model(
    model, 
    tokenizer, 
    train_dataset, 
    output_dir="qwen_finetuned_model",
    max_seq_length=2048,
    num_train_epochs=1,
    max_steps=60,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
):
    """
    使用 LoRA 对模型进行微调
    """
    print("=" * 60)
    print("开始配置 LoRA 模型...")
    print("=" * 60)
    
    # 1. 配置 LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,  # LoRA rank，建议：8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,  # 通常设置为 rank 的 1-2 倍
        lora_dropout = 0,  # 0 表示不使用 dropout（优化性能）
        bias = "none",  # "none" 表示不训练 bias（优化性能）
        use_gradient_checkpointing = "unsloth",  # 使用 unsloth 优化，节省 30% 显存
        random_state = 3407,
        use_rslora = False,  # Rank Stabilized LoRA
        loftq_config = None,  # LoftQ 配置
    )
    
    print("✓ LoRA 配置完成")
    
    # 2. 转换数据集为 SFT 格式
    print("\n" + "=" * 60)
    print("准备训练数据...")
    print("=" * 60)
    train_dataset_sft = convert_math_to_sft_format(train_dataset)
    print(f"✓ 训练数据准备完成，共 {len(train_dataset_sft)} 条样本")
    
    # 3. 配置训练器
    print("\n" + "=" * 60)
    print("配置训练器...")
    print("=" * 60)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset_sft,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,  # 对于长序列，packing 可以提速 5 倍
        args=SFTConfig(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,  # 有效 batch size = 2 * 4 = 8
            warmup_steps=5,
            num_train_epochs=num_train_epochs,  # 完整训练轮数
            max_steps=max_steps,  # 最大训练步数（优先级高于 num_train_epochs）
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",  # 8-bit Adam 优化器
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="none",  # 可以改为 "wandb" 或 "tensorboard"
        ),
    )
    
    print("✓ 训练器配置完成")
    print(f"  - Batch size: {per_device_train_batch_size}")
    print(f"  - Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  - Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Max steps: {max_steps}")
    
    # 4. 开始训练
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    trainer_stats = trainer.train()
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"训练统计: {trainer_stats}")
    
    # 5. 保存模型
    print("\n" + "=" * 60)
    print("保存微调后的模型...")
    print("=" * 60)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✓ 模型已保存到: {output_dir}")
    
    return model, tokenizer


def main():
    print("\n" + "=" * 60)
    print("加载数据集...")
    print("=" * 60)
    
    train_dataset = load_questions("train")
    test_dataset = load_questions("test")
    
    print(f"✓ 训练集: {len(train_dataset)} 条")
    print(f"✓ 测试集: {len(test_dataset)} 条")
    
    print("\n" + "=" * 60)
    print("加载基础模型...")
    print("=" * 60)
    
    qwen_model, qwen_tokenizer = load_model(model_name="./Qwen3-0.6B")
    print("✓ 模型加载完成")
    
    # ===== 选项 1: 评估基础模型（未微调） =====
    print("\n" + "=" * 60)
    print("评估基础模型性能...")
    print("=" * 60)
    
    num_few_shot_gold = 3
    gold_examples = construct_gold_few_shot_examples(train_dataset, num_few_shot_gold)
    
    model_name, acc = evaluate_model_performance(
        test_dataset=test_dataset,
        few_shot_examples=gold_examples,
        model_name="qwen3-0.6B (baseline)",
        model=qwen_model,
        tokenizer=qwen_tokenizer,
        device="cuda"
    )
    print(f"\n{'=' * 60}")
    print(f"基础模型: {model_name}")
    print(f"准确率: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'=' * 60}")
    
    # ===== 选项 2: 微调模型（取消注释以启用） =====
    # print("\n" + "=" * 60)
    # print("开始微调...")
    # print("=" * 60)
    # 
    # finetuned_model, finetuned_tokenizer = finetune_model(
    #     model=qwen_model,
    #     tokenizer=qwen_tokenizer,
    #     train_dataset=train_dataset,
    #     output_dir="qwen_finetuned_model",
    #     max_seq_length=2048,
    #     max_steps=60,
    #     per_device_train_batch_size=2,
    #     gradient_accumulation_steps=4,
    #     learning_rate=2e-4,
    # )
    # 
    # # 评估微调后的模型
    # print("\n" + "=" * 60)
    # print("评估微调后的模型性能...")
    # print("=" * 60)
    # 
    # FastLanguageModel.for_inference(finetuned_model)  # 启用推理模式
    # 
    # model_name, acc = evaluate_model_performance(
    #     test_dataset=test_dataset,
    #     few_shot_examples=gold_examples,
    #     model_name="qwen3-0.6B (finetuned)",
    #     model=finetuned_model,
    #     tokenizer=finetuned_tokenizer,
    #     device="cuda"
    # )
    # print(f"\n{'=' * 60}")
    # print(f"微调后模型: {model_name}")
    # print(f"准确率: {acc:.4f} ({acc*100:.2f}%)")
    # print(f"{'=' * 60}")

if __name__ == "__main__":
    main()