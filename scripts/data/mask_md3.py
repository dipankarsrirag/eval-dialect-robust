import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeds(sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

def get_similarity(embed_1, embed_2):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(embed_1, embed_2)[0].item()

def clean(temp):
    def replace(x):
        x = x.replace("\n\n", "\n")
        x = x.replace(",", "")
        x = x.replace("*", "")
        return x
    temp["transcript"] = temp["transcript"].apply(lambda x : replace(x))
    return temp

def mask(guesser, target):
    found = False
    for i, turn in enumerate(guesser):
        if target.lower() in turn.lower():
            guesser[i] = "Guesser: [MASK]"
            found = True
            break
    if not found:
        target_embed = get_embeds(target)
        
        turn_embeds = [get_embeds(turn.split(":")[1]) for turn in guesser]
        
        sim_scores = [get_similarity(target_embed, embed) for embed in turn_embeds]
        i = np.argmax(sim_scores)
        guesser[i] = "Guesser: [MASK]"
    return guesser[:i+1]

def process(transcript, target):
    turns = transcript.split("\n")
    describer = turns[::2]
    if len(turns) == 1: #The guesser is either missing, or the conversation was not created properly.
        if ":" not in describer[0]:
            describer[0] = "Describer: "+describer[0]
        guesser = mask(describer, target)
    else:
        guesser = mask(turns[1::2], target)
    
    transcript = []
    while len(guesser) > 0:
        transcript.append(describer.pop(0))
        transcript.append(guesser.pop(0))
    return '\n'.join(transcript)

if __name__ == "__main__":
    
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    subsets = ["ind_eng", "us_eng", "ai_trans", "ai_gen"]

    for subset in subsets:
        data = pd.read_json(f"./md3/cleaned_data/{subset}.jsonl")
        data = clean(data.copy())
        for i in range(len(data)):
            transcript = data.iloc[i]["transcript"]
            target = data.iloc[i]["target_word"]
        
            data.loc[i, "transcript"] = process(transcript, target)
    
        # data = data[["transcript", "target_word", "restricted_words"]]
        data.to_json(f"./m-md3/{subset}.jsonl", orient="records")