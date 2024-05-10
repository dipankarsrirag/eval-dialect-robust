import pandas as pd
import numpy as np

if __name__ == "__main__":
    
    sets = ["ind_eng", "us_eng", "ai_trans", "ai_gen"]
    for set in sets:
        data = pd.read_json(f"./m-md3/{set}.jsonl")
        ids = np.arange(len(data))
        np.random.shuffle(ids)

        train_size = int(len(ids) * 0.15)
        valid_size = int(len(ids) * 0.10)
        test_size = len(ids) - train_size - valid_size
        
        train_ids = ids[:train_size]
        valid_ids = ids[train_size : train_size + valid_size]
        test_ids = ids[train_size + valid_size :]

        train_data = data.iloc[train_ids].reset_index(drop=True)
        valid_data = data.iloc[valid_ids].reset_index(drop=True)
        test_data = data.iloc[test_ids].reset_index(drop=True)
        
        train_data.to_json(f"./m-md3/train/{set}.jsonl", orient="records")
        valid_data.to_json(f"./m-md3/valid/{set}.jsonl", orient="records")
        test_data.to_json(f"./m-md3/test/{set}.jsonl", orient="records")