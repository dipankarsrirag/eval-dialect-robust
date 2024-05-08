"""
Refer to Section 2 of the paper for more information.
We prompt gpt-4-turbo-preview to remove dialectic information in IndEng.
The resultant dataset is known as AITrans. We use this program ot create the AITrans subset.
"""

from dotenv import load_dotenv
from openai import OpenAI
import os
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    # Load all environment variables from .env.
    load_dotenv()

    # OpenAI API key.
    api_key = os.getenv("OPEN_AI_KEY")

    # IndEng conversations from MD3.
    ind_eng = pd.read_json("./md3/cleaned_data/ind_eng.jsonl")

    client = OpenAI(api_key=api_key)
    neutral = []
    for transcript in tqdm(list(ind_eng["transcript"])):
        completion = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {  # Prompt used to translate conversations in IndEng to AITrans dialect.
                    "role": "system",
                    "content": "Normalise the conversation. Remove all exaggerations and dialectal information. Return a neutral response.",
                },
                {"role": "user", "content": transcript},
            ],
            # Lower temperature for reduced stochasticity and better reproducibility.
            temperature=0.1,
        )
        neutral.append(completion.choices[0].message.content)

    ai_trans = ind_eng.copy()
    ai_trans["transcript"] = neutral
    ai_trans.to_json("./md3/cleaned_data/ai_trans.jsonl", orient="records")
