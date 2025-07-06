import utils
from openai import OpenAI
import dotenv
import os
import json

def send(r, N, i0):
    dotenv.load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    print(api_key)
    client = OpenAI(api_key=api_key)
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    utils.append_generation_csvs(output_dir='moran_process_output', output_file='all_generations.csv')
    tasks = []
    for i in range(0,5):
        task = utils.construct_task_batch(model_name='gpt-4o-mini')
        tasks.append(task)
    utils.generate_jsonl(tasks, 'batch.jsonl')
    batch_file = utils.send_batch(client, "batch.jsonl")
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(batch_job)
    # Write the batch job id to a file
    with open(f"batch_job_ids_{model_name}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"batch_job_id": batch_job.id, "r": r, "N": N, "i0": i0}) + "\n")
# def main():
#     send()

# if __name__ == "__main__":
#     main()