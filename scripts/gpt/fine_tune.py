import datetime
from openai import OpenAI
import pickle as pk
import signal
import time
from tqdm import tqdm
from dotenv import load_dotenv
import os

def fine_tune(ids, subset, choice=False):

    def signal_handler(sig, frame):
        status = client.fine_tuning.jobs.retrieve(job_id).status
        print(f"Stream interrupted. Job is still {status}.")
        return

    train = ids["tws" if choice else "twp"]["train"]
    valid = ids["tws" if choice else "twp"]["valid"]

    print(
        f"Fine-tuning: gpt-3.5-turbo-0125 | {subset} | {'TWS' if choice else 'TWP'}"
    )
    response = client.fine_tuning.jobs.create(
        training_file=ids[train],
        validation_file=ids[valid],
        model="gpt-3.5-turbo-0125",
        hyperparameters={
            "n_epochs": 20,
            "learning_rate_multiplier": 0.3,
            "batch_size": 3,
        },
        suffix=f"tws-{subset}-0125" if choice else f"twp-{subset}-0125",
    )

    job_id = response.id
    status = client.fine_tuning.jobs.retrieve(job_id).status

    if status not in ["succeeded", "failed"]:
        print(f"Job not in terminal status: {status}. Waiting.")
        while status not in ["succeeded", "failed"]:
            time.sleep(10)
            status = client.fine_tuning.jobs.retrieve(job_id).status
            if status == "queued":
                time.sleep(50)
                continue
            print(f"Status: {status}")
    else:
        print(f"Finetune job {job_id} finished with status: {status}")
    print(
        f"Streaming events for the fine-tuning model: {response.model} | {subset} | {'TWS' if choice else 'TWP'}"
    )

    signal.signal(signal.SIGINT, signal_handler)

    events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id)
    try:
        for event in events:
            print(
                f"{datetime.datetime.fromtimestamp(event.created_at)} {event.message}"
            )
    except Exception:
        print("Stream interrupted (client disconnected).")

    model = client.fine_tuning.jobs.retrieve(job_id).fine_tuned_model

    return model


if __name__ == "__main__":
    
    load_dotenv()
    # OpenAI API key.
    api_key = os.getenv("OPEN_AI_KEY")
    client = OpenAI(api_key=api_key)

    with open("./messages/files.pk", "rb") as f:
        files = pk.load(f)

    sets = ["ind_eng", "us_eng", "ai_trans", "ai_gen"]

    models = {}

    for subset in sets:
        ids = files[subset]
        temp = {}
        for choice in [True, False]:
            model = fine_tune(ids, subset, choice)
            if choice:
                temp["tws"] = model
            else:
                temp["twp"] = model
        models[subset] = temp

    with open("./models/gpt_3_ids.pk", "wb") as f:
        pk.dump(models, f)
