import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import os
from openai import OpenAI
import pickle as pk
import json

if __name__ == "__main__":
    
    load_dotenv()

    # OpenAI API key.
    api_key = os.getenv("OPEN_AI_KEY")
    client = OpenAI(api_key=api_key)
    
    files = dict()
    
    sets = ["ind_eng", "us_eng", "ai_trans", "ai_gen"]
    for subset in sets:
        words = set(pd.read_json(f"./m-md3/{subset}.jsonl")["target_word"])
        train = pd.read_json(f"./m-md3/train/{subset}.jsonl")
        valid = pd.read_json(f"./m-md3/valid/{subset}.jsonl")
        
        train_X, train_y = train["transcript"], train["target_word"]
        valid_X, valid_y = valid["transcript"], valid["target_word"]
        
        messages = []
        for input, output in tqdm(zip(train_X, train_y)):
            message = {
                "messages": [
                    {
                        "role": "system",
                        "content": 'In the conversation, fill the "[MASK]" with only one relevant noun or verb.',
                    },
                    {"role": "user", "content": f'Conversation: "{input}"'},
                    {"role": "assistant", "content": f"Word: {output}"},
                ]
            }
            messages.append(message)
        
        with open(f"./messages/train/{subset}_twp.jsonl", "w") as f:
            for message in messages:
                f.write(json.dumps(message) + "\n")
        
        messages = []    
        for input, output in tqdm(zip(valid_X, valid_y)):
            message = {
                "messages": [
                    {
                        "role": "system",
                        "content": 'In the conversation, fill the "[MASK]" with only one relevant noun or verb.',
                    },
                    {"role": "user", "content": f'Conversation: "{input}"'},
                    {"role": "assistant", "content": f"Word: {output}"},
                ]
            }
            messages.append(message)
        
        with open(f"./messages/valid/{subset}_twp.jsonl", "w") as f:
            for message in messages:
                f.write(json.dumps(message) + "\n")
        
        messages = []
        for input, output in tqdm(zip(train_X, train_y)):
            message = {
                "messages": [
                    {
                        "role": "system",
                        "content": f'In the conversation, choose a word from the list below to fill the "[MASK]". List of choices: {words}',
                    },
                    {"role": "user", "content": f'Conversation: "{input}"'},
                    {"role": "assistant", "content": f"Word: {output}"},
                ]
            }
            messages.append(message)
        
        with open(f"./messages/train/{subset}_tws.jsonl", "w") as f:
            for message in messages:
                f.write(json.dumps(message) + "\n")
        
        messages = []    
        for input, output in tqdm(zip(valid_X, valid_y)):
            message = {
                "messages": [
                    {
                        "role": "system",
                        "content": f'In the conversation, choose a word from the list below to fill the "[MASK]". List of choices: {words}',
                    },
                    {"role": "user", "content": f'Conversation: "{input}"'},
                    {"role": "assistant", "content": f"Word: {output}"},
                ]
            }
            messages.append(message)
        
        with open(f"./messages/valid/{subset}_tws.jsonl", "w") as f:
            for message in messages:
                f.write(json.dumps(message) + "\n")
                

        twp_train = client.files.create(
            file=open(f"./messages/train/{subset}_twp.jsonl", "rb"), purpose="fine-tune"
        )

        twp_valid = client.files.create(
            file=open(f"./messages/valid/{subset}_twp.jsonl", "rb"), purpose="fine-tune"
        )

        tws_train = client.files.create(
            file=open(f"./messages/train/{subset}_tws.jsonl", "rb"),
            purpose="fine-tune",
        )

        tws_valid = client.files.create(
            file=open(f"./messages/valid/{subset}_tws.jsonl", "rb"),
            purpose="fine-tune",
        )

        files[subset] = {
            "twp": 
                {
                    "train": twp_train, "valid": twp_valid
                    }, 
            "tws": {
                "train": tws_train, "valid": tws_valid
                }
            }
        
    with open("./messages/files.pk", "wb") as f:
        pk.dump(files, f)
        
        