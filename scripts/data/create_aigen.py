"""
Refer to Section 2 of the paper for more information.
We use generated target and restricted words to prompt gpt-4-turbo-preview and obtain synthetic dialogues that constitute AIGen.
We use this program ot create the AIGen subset.
"""

from dotenv import load_dotenv
from openai import OpenAI
import os
import pandas as pd
import random
from tqdm import tqdm

if __name__ == "__main__":
    data = pd.read_json("./md3/raw_data/prompts_ai_gen.jsonl", lines=True)
    word = []

    load_dotenv()

    # OpenAI API key.
    api_key = os.getenv("OPEN_AI_KEY")
    client = OpenAI(api_key=api_key)
    
    ai_gen = {
        "transcript": [],
        "target_word": [],
        "restricted_words": [],
    }

    transcript = []
    for i in tqdm(range(len(data))):
        for _ in range(random.randint(1, 2)):
            completion = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": 'Given a target word, the "Describer" provides short descriptions to the "Guesser", while also not using the restricted words. "Guesser" asks questions to get more hints from "Describer". The "Describer" always starts the conversation. Generate the conversation.',
                    },
                    {
                        "role": "user",
                        "content": f'Target word: "{data["target_word"].iloc[i]}". Restricted words: {data["restricted_words"].iloc[i]}',
                    },
                ],
            )
            ai_gen["transcript"].append(completion.choices[0].message.content)
            ai_gen["target_word"].append(data["target_word"].iloc[i])
            ai_gen["restricted_words"].append(data["restricted_words"].iloc[i])
    
    ai_gen = pd.DataFrame(ai_gen)
    ai_gen.to_json("./md3/cleaned_data/ai_gen.jsonl", orient="records")
