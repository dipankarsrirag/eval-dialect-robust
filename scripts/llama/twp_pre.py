import os
from tqdm import tqdm
import pathlib
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse

import pandas as pd

import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICE"] = "0,1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
dtype = torch.bfloat16


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str)
    parser.add_argument("out_path", type=pathlib.Path)
    parser.add_argument("locale", type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    data = pd.read_json(args.test_path)
    gens = []
    for text in tqdm(list(data["transcript"]), desc="Generating"):
        messages = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nFrom the conversation, Replace "[MASK]" with the most relevant word. Generate the word, do not give explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>\nConversation:{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        encodeds = tokenizer(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)
        generated_ids = model.generate(
            input_ids=encodeds["input_ids"],
            attention_mask=encodeds["attention_mask"],
            max_length=encodeds["input_ids"].shape[1] + 10,
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

    data[f"twp_pre_{args.locale}"] = gens
    data.to_json(args.out_path, orient="records")


if __name__ == "__main__":
    main()
