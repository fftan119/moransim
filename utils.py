import os
import json
import csv
import datetime
import time


system_prompt = r'''
The .csv file is a discrete history of mutant A and individuals B within a population of 20 following the Moran process. The format per evolution step is [index of reproducing individual][individual type]:[index of dying individual][individual type]. The .csv file is a entire evolution history with the very last state being an absorbing state. Recover the total number of intial mutants, and relative fitness. Give me exact values.

Moran Process


The \textit{Moran process} is a foundational stochastic model that describes the evolution of a finite, well-mixed population undergoing reproduction and death in discrete time steps \cite{moran1958random,Nowak2006}. The population size remains constant at (N), consisting of two competing types: mutants ((A)) with relative fitness (r), and residents ((B)) with fitness (1). At any given generation, let (i \in {0, 1, \ldots, N}) represent the number of mutants in the population.

Each generational update proceeds as follows:
\begin{itemize}
    \item \textbf{Reproduction:} One individual is chosen to reproduce, with probability proportional to fitness:
    [
        \Pr(\text{birth of }A) = \frac{r i}{r i + (N - i)}, \qquad
        \Pr(\text{birth of }B) = \frac{N - i}{r i + (N - i)}.
    ]

    \item \textbf{Death:} Independently, one individual is selected uniformly at random to be removed:
    [
        \Pr(\text{death of }A) = \frac{i}{N}, \qquad
        \Pr(\text{death of }B) = \frac{N - i}{N}.
    ]
\end{itemize}

The combination of these two events defines a Markov chain with the following transition probabilities:

    \Pr(i \rightarrow i + 1) &= \frac{r i}{r i + (N - i)} \cdot \frac{N - i}{N}, \
    \Pr(i \rightarrow i - 1) &= \frac{N - i}{r i + (N - i)} \cdot \frac{i}{N}, \
    \Pr(i \rightarrow i) &= 1 - \Pr(i \rightarrow i+1) - \Pr(i \rightarrow i-1).


The process has two absorbing states: (i = 0) (extinction of mutants) and (i = N) (fixation of mutants).
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
    Returns the system prompt for Moran process parameter inference.
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
