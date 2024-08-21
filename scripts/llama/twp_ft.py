import os

os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICE"] = "0,1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from tqdm import tqdm

import pathlib
import argparse

import torch

import pandas as pd
from datasets import Dataset

import bitsandbytes as bnb
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str)
    parser.add_argument("locale", type=str)

    parser.add_argument("train_path", type=pathlib.Path)
    parser.add_argument("valid_path", type=pathlib.Path)
    parser.add_argument("out_path", type=pathlib.Path)

    parser.add_argument("lora_alpha", type=int)
    parser.add_argument("lora_r", type=int)
    parser.add_argument("lora_dropout", type=float)

    parser.add_argument("cache_dir", type=pathlib.Path)
    parser.add_argument("num_epochs", type=int)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("learning_rate", type=float)

    parser.add_argument("adapter_dir", type=pathlib.Path)

    args = parser.parse_args()

    return args


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if "lm_head" in lora_module_names:  # needed for 16-bit
            lora_module_names.remove("lm_head")
    return list(lora_module_names)


def main():
    
    def get_prompt(data):
        prompts = []
        for text, word in zip(data["transcript"], data["target_word"]):
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nFrom the conversation, Replace "[MASK]" with the most relevant word. Generate the word, do not give explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>\nConversation:{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{word}<|eot_id|>"""
            prompts.append(prompt)
        return prompts
    
    args = parse_args()

    train = pd.read_json(args.train_path)
    valid = pd.read_json(args.valid_path)

    train["prompt"] = get_prompt(train)
    valid["prompt"] = get_prompt(valid)

    train = Dataset.from_pandas(train[["prompt"]], split="train")
    valid = Dataset.from_pandas(valid[["prompt"]], split="test")

    model_id = args.model_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    modules = find_all_linear_names(model)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        target_modules=modules,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=args.cache_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_steps=0.3,
        learning_rate=args.learning_rate,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=valid,
        dataset_text_field="prompt",
        peft_config=peft_config,
        max_seq_length=2048,
        tokenizer=tokenizer,
    )

    trainer.train()

    adaptor_dir = args.adapter_dir

    trainer.model.save_pretrained(adaptor_dir)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    merged_model = PeftModel.from_pretrained(base_model, adaptor_dir)

    data = pd.read_json(args.out_path)
    gens = []
    for text in tqdm(list(data["transcript"]), desc="Generating"):
        messages = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nFrom the conversation, Replace "[MASK]" with the most relevant word. Generate the word, do not give explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>\nConversation:{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        encodeds = tokenizer(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)
        generated_ids = merged_model.generate(
            input_ids=encodeds["input_ids"],
            attention_mask=encodeds["attention_mask"],
            max_length=encodeds["input_ids"].shape[1] + 4,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(generated_ids)
        word = (
            decoded[0]
            .split("<|start_header_id|>assistant<|end_header_id|>", 1)[1]
            .split("<|eot_id|>", 1)[0]
            .strip()
        )
        gens.append(word)
        print(word)
        torch.cuda.empty_cache()
        del model_inputs
        del generated_ids
        del decoded
        del encodeds
        del messages
        del word

    data[f"twp_ft_{args.locale}"] = gens
    data.to_json(args.out_path, orient="records")
