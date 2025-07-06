import os
import json
import csv
import datetime
import time

system_prompt = '''
Your goal is to figure out the correct parameters i0, r, N for the Moran process simulation based on the simulation data provided.
Iterations (generations) per simulation are formatted as follows:
1A:2B
3A:4A
5B:6B
Where the first number is the index of the individual that was born, followed by its type (A or B), then a colon, and finally the index of the individual that died, followed by its type.

Neutral generations are where the number of A and B individuals remains the same, while mutant generations are where the number of A/B individuals increases.
You will be given a collection of simulations, and you must determine the parameters (i0, r, N) that best fit the data.
'''
system_prompt = '''
The .csv file is a discrete history of mutant A and individuals B within a population of 20 following the Moran process. The format per evolution step is [index of reproducing individual][individual type]:[index of dying individual][individual type]. The .csv file is a entire evolution history with the very last state being an absorbing state. Recover the total number of intial mutants, and relative fitness. Give me exact values.
You will be given a collection of simulations, and you must determine the parameters (i0, r, N) that best fit the data.
i0 is a positive integer representing the initial number of mutants in the population.
r is a float representing the relative fitness of the mutants compared to the wild type. Round to the nearest tenth.
N is a positive integer representing the total population size.

Please respond in JSON format with the following structure:
{
  "i0": <initial_mutants>,
  "r": <relative_fitness>,
  "N": <population_size>
}
'''
def append_generation_csvs(output_dir='moran_process_output', output_file='all_generations.csv'):
    csv_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    if not csv_files:
        print("No CSV files found.")
        return

    header_written = False
    output_path = os.path.join(output_dir, output_file)
    # Always overwrite the old all_generations.csv file
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = None
        for csv_file in csv_files:
            with open(csv_file, 'r', newline='', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                try:
                    header = next(reader)
                except StopIteration:
                    continue  # skip empty files
                if not header_written:
                    writer = csv.writer(outfile)
                    writer.writerow(['source_file'] + header)
                    header_written = True
                for row in reader:
                    writer.writerow([os.path.basename(csv_file)] + row)

def get_system_prompt():
    """
    Returns the system prompt for the physics multiple choice question answering task.
    """
    return system_prompt.strip()

def generate_jsonl(tasks, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')

def construct_task_batch(model_name='gpt-4o-mini'):
    with open('moran_process_output/all_generations.csv', 'r', encoding='utf-8') as f:
        csv_content = f.read()
    return {
        "custom_id": f"task-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{int(time.time() * 1e6)}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "temperature": 0.1,
            "response_format": {
                "type": "json_object"
            },
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": csv_content
                }
            ],
        }
    }

def send_batch(client, file_name='batch.jsonl'):
    batch_file = client.files.create(
        file=open(file_name, 'rb'),
        purpose='batch'
    )
    print(batch_file)
    return batch_file