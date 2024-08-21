from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
import argparse
import pathlib
import torch

warnings.filterwarnings("ignore")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str)
    parser.add_argument("cache_path", type=str)
    parser.add_argument("save_path", type=pathlib.Path)
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
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
        cache_dir=args.cache_path,
        quantization_config=bnb_config if device == "cuda" else None,
    )
    print(model)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=args.cache_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(["[MASK]"])
    tokenizer.mask_token = "[MASK]"
    print(tokenizer)

    model.resize_token_embeddings(len(tokenizer))

    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    print(model)
    print(tokenizer)


if __name__ == "__main__":
    main()
