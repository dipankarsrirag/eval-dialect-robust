import json
import os
import pickle as pk

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

def get_prompt(transcript, words, task):
    if task == "twp":
        return [
            {
                "role": "system",
                "content": "From the conversation, Replace [MASK] with the most relevant word. Generate the word, do not give explanation.",
            },
            {"role": "user", "content": f"Conversation: {transcript}"},
        ]
    else:
        return [
            {
                "role": "system",
                "content": "From the conversation, Replace [MASK] with the most relevant word from the list of target words. Generate the word, Do not give explanation.",
            },
            {
                "role": "user",
                "content": f"Conversation: {transcript}\nTarget Words: {words}",
            },
        ]


if __name__ == "__main__":

    load_dotenv()

    # OpenAI API key.
    api_key = os.getenv("OPEN_AI_KEY")
    client = OpenAI(api_key=api_key)

    files = dict()

    sets = ["ind_eng", "us_eng", "ai_trans", "ai_eng", "multi"]
    for subset in sets:
        train = pd.read_json(f"./data/train/{subset}.jsonl")
        valid = pd.read_json(f"./data/valid/{subset}.jsonl")
        test = pd.read_json(f"./data/test/{subset}.jsonl")
        words = set(
            list(train["target_word"])
            + list(valid["target_word"])
            + list(test["target_word"])
        )

        train_X, train_y = train["transcript"], train["target_word"]
        valid_X, valid_y = valid["transcript"], valid["target_word"]

        messages = []
        for input, output in tqdm(zip(train_X, train_y)):
            message = {
                "messages": [
                    {
                        "role": "system",
                        "content": "From the conversation, Replace [MASK] with the most relevant word. Generate the word, do not give explanation..",
                    },
                    {"role": "user", "content": f'Conversation: "{input}"'},
                    {"role": "assistant", "content": f"{output}"},
                ]
            }
            messages.append(message)

        with open(f"./messages/gpt/train/{subset}_twp.jsonl", "w") as f:
            for message in messages:
                f.write(json.dumps(message) + "\n")

        messages = []
        for input, output in tqdm(zip(valid_X, valid_y)):
            message = {
                "messages": [
                    {
                        "role": "system",
                        "content": "From the conversation, Replace [MASK] with the most relevant word. Generate the word, do not give explanation.",
                    },
                    {"role": "user", "content": f'Conversation: "{input}"'},
                    {"role": "assistant", "content": f"{output}"},
                ]
            }
            messages.append(message)

        with open(f"./messages/gpt/valid/{subset}_twp.jsonl", "w") as f:
            for message in messages:
                f.write(json.dumps(message) + "\n")

        messages = []
        for input, output in tqdm(zip(train_X, train_y)):
            message = {
                "messages": [
                    {
                        "role": "system",
                        "content": "From the conversation, Replace [MASK] with the most relevant word from the list of target words. Generate the word, Do not give explanation.",
                    },
                    {
                        "role": "user",
                        "content": f"Conversation: {input}\nTarget Words: {words}",
                    },
                    {"role": "assistant", "content": f"{output}"},
                ]
            }
            messages.append(message)

        with open(f"./messages/gpt/train/{subset}_tws.jsonl", "w") as f:
            for message in messages:
                f.write(json.dumps(message) + "\n")

        messages = []
        for input, output in tqdm(zip(valid_X, valid_y)):
            message = {
                "messages": [
                    {
                        "role": "system",
                        "content": "From the conversation, Replace [MASK] with the most relevant word from the list of target words. Generate the word, Do not give explanation.",
                    },
                    {
                        "role": "user",
                        "content": f"Conversation: {input}\nTarget Words: {words}",
                    },
                    {"role": "assistant", "content": f"{output}"},
                ]
            }
            messages.append(message)

        with open(f"./messages/gpt/valid/{subset}_tws.jsonl", "w") as f:
            for message in messages:
                f.write(json.dumps(message) + "\n")

        twp_train = client.files.create(
            file=open(f"./messages/gpt/train/{subset}_twp.jsonl", "rb"),
            purpose="fine-tune",
        )

        twp_valid = client.files.create(
            file=open(f"./messages/gpt/valid/{subset}_twp.jsonl", "rb"),
            purpose="fine-tune",
        )

        tws_train = client.files.create(
            file=open(f"./messages/gpt/train/{subset}_tws.jsonl", "rb"),
            purpose="fine-tune",
        )

        tws_valid = client.files.create(
            file=open(f"./messages/gpt/valid/{subset}_tws.jsonl", "rb"),
            purpose="fine-tune",
        )

        files[subset] = {
            "twp": {"train": twp_train, "valid": twp_valid},
            "tws": {"train": tws_train, "valid": tws_valid},
        }

    with open("./messages/gpt/files.pk", "wb") as f:
        pk.dump(files, f)
