"""
We use this program to process the conversations from MD3 in Indian English and US English. Refer to Section 2 in the paper.
"""

import pandas as pd
import ast
from tqdm import tqdm


def group_transcript(transcript):
    """

    Args:
        transcript (pd.DataFrame): Dataframe containing the utterances by both the players participating in a single instance of dialogue word game-- Taboo.

    Returns:
        str: Processed transcript of the single instance of a dialogue word game-- Taboo.
    """
    speakers = {0: [], 1: []}
    dial = ""
    i = 0
    while i < len(transcript):
        if transcript["speaker_id"].iloc[i] == "s0" and len(speakers[1]) == 0:
            while i < len(transcript) and transcript["speaker_id"].iloc[i] == "s0":
                speakers[0].append(transcript["transcript"].iloc[i])
                i += 1

            dial = dial + "s0: " + ". ".join(speakers[0]) + "\n"
            speakers[0] = []
        else:
            while i < len(transcript) and transcript["speaker_id"].iloc[i] != "s0":
                speakers[1].append(transcript["transcript"].iloc[i])
                i += 1

            dial = dial + "s1: " + ". ".join(speakers[1]).strip() + "\n"
            speakers[1] = []

    dial = dial.strip()
    turns = len(dial.split("\n"))
    desc = dial.split("\n")[0].split(" ")[0]
    if turns > 1:
        gues = dial.split("\n")[1].split(" ")[0]
        dial = dial.replace(gues, "Guesser:")
    dial = dial.replace(desc, "Describer:")
    dial = dial.replace("?.", "?")
    dial = dial.replace("!.", "!")
    dial = dial.replace("..", ".")
    dial = dial.replace(",", "")
    dial = dial.replace(" .", ".")

    return dial.strip()


if __name__ == "__main__":
    dialects = ["ng"]

    for dial in dialects:
        # For each human generated dialect.
        transcripts = pd.read_csv(f"./md3/raw_data/transcripts_en_{dial}.tsv", sep="\t")
        speakers = pd.read_csv(f"./md3/raw_data/speakers_en_{dial}.tsv", sep="\t")
        prompts = pd.read_csv(f"./md3/raw_data/prompts_en_{dial}.tsv", sep="\t")

        # Filtering the dataset to include only the word games where the guesser correctly identified the target word.
        word_win_prompts = prompts.loc[
            prompts["game_type"].isin(["word"]) & prompts["prompt_status"].isin(["win"])
        ].reset_index(drop=True)[
            ["clip_identifier", "correct_word/image", "distractors"]
        ]

        word_win_ids = list(word_win_prompts["clip_identifier"])
        word_win_transcripts = transcripts.loc[
            transcripts["clip_identifier"].isin(word_win_ids)
        ].reset_index(drop=True)[
            ["clip_identifier", "transcript", "speaker_id", "match_id", "round"]
        ]

        word_win_transcripts.iloc[:, 2] = word_win_transcripts.iloc[:, 2].apply(
            lambda x: f"s{int(x[7:])}"
        )
        word_win_transcripts.dropna(inplace=True)
        word_win_transcripts["describer"] = word_win_transcripts[
            "clip_identifier"
        ].apply(lambda x: x.split("_")[2])

        data = {
            "clip_id": [],
            "transcript": [],
            "target_word": [],
            "restricted_words": [],
            "round": [],
        }

        # Processing the word game transcripts to look like a structured dialogue.
        for i in tqdm(set(word_win_transcripts["clip_identifier"])):
            transcript = (
                word_win_transcripts.loc[
                    word_win_transcripts["clip_identifier"].isin([i])
                ]
                .reset_index(drop=True)[["transcript", "speaker_id", "round"]]
                .copy()
            )
            data["transcript"].append(group_transcript(transcript))
            temp = word_win_prompts.loc[
                word_win_prompts["clip_identifier"].isin([i])
            ].reset_index(drop=True)[["correct_word/image", "distractors"]]
            target_word = temp.iloc[0, 0]
            restricted_words = ", ".join(ast.literal_eval(temp.iloc[0, 1]))  # type: ignore
            data["target_word"].append(target_word)
            data["restricted_words"].append(restricted_words)
            data["clip_id"].append(i)
            data["round"].append(transcript["round"].iloc[0])

        data = pd.DataFrame(data)
        data.to_json(f"./md3/cleaned_data/{dial}_eng.jsonl", orient="records")
