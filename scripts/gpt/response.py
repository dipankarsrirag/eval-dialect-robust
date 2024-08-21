from openai import OpenAI
import os
import pickle as pk
import pandas as pd
import sys
from tqdm import tqdm
from dotenv import load_dotenv


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


if len(sys.argv) < 3:
    sys.exit(-1)

subset = sys.argv[1]
task = sys.argv[2]

load_dotenv()
with open("./models/gpt/multi_ids.pk", "rb") as f:
    models = pk.load(f)


models[subset][task].extend(["gpt-3.5-turbo-0125", "gpt-4-turbo-preview"])
models = models[subset][task]


api_key = os.getenv("OPEN_AI_KEY")
client = OpenAI(api_key=api_key)

print(models)

train = pd.read_json(f"./data/train/{subset}.jsonl")
valid = pd.read_json(f"./data/valid/{subset}.jsonl")
test = pd.read_json(f"./outputs/{subset}.jsonl", encoding="utf-8")
words = set(
    list(train["target_word"]) + list(valid["target_word"]) + list(test["target_word"])
)

for model in models:
    responses = []
    for i in tqdm(range(len(test)), desc=f"Model: {model} | Task: {task}"):
        transcript = test.iloc[i].transcript
        message = get_prompt(transcript=transcript, words=words, task=task)
        response = (
            client.chat.completions.create(model=model, messages=message, max_tokens=5)
            .choices[0]
            .message.content
        )
        responses.append(response)
    if model.split(":")[0] == "ft":
        test[f"gpt_3.5_ft_{task}"] = responses
    elif "gpt-4" in model:
        test[f"gpt4_pre_{task}"] = responses
    else:
        test[f"gpt3.5_pre_{task}"] = responses

test.to_json(f"./outputs/{subset}.jsonl", orient="records")
