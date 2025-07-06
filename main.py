import csv
import os
import random
import shutil
from send import send  # Import the send function from send.py

def moran_process(r, N, i0, gen_num):
    """
    Moran process with fitness-proportional reproduction and uniform death.
    Outputs only birth-death events per generation in the form [birth_index][type]:[death_index][type].
    No header row is included in the output CSV.
    """
    assert 0 < i0 < N, "Initial mutant count must be between 1 and N-1"

    i = i0
    population = ['A'] * i + ['B'] * (N - i)

    event_data = []  # List of [event_string]

    # Output directory
    output_dir = "moran_process_output"
    os.makedirs(output_dir, exist_ok=True)

    # Handle auto-increment filename
    base_name = f"Generation {gen_num}"
    file_path = os.path.join(output_dir, f"{base_name}.csv")
    # No suffix needed; always overwrite the file if it exists

    while 0 < i < N:
        # Fitness-proportional birth
        weights = [r if ind == 'A' else 1 for ind in population]
        birth_index = random.choices(range(N), weights=weights)[0]
        birth_type = population[birth_index]

        # Uniform random death
        death_index = random.randrange(N)
        death_type = population[death_index]

        # Replace the individual
        population[death_index] = birth_type

        # Update mutant count
        i = population.count('A')

        # Store the event string
        event_data.append([f"{birth_index}{birth_type}:{death_index}{death_type}"])

    # Write to CSV (no header)
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(event_data)

    print(f"Simulation complete. Saved to: {file_path}")


# Example usage
if __name__ == "__main__":
    for j in range(1, 11):
        r = random.randint(10, 15)/10  # Random relative fitness between 1.0 and 1.5
        N = random.randint(15, 25)
        i0 = random.randint(1, N-1)  # Random initial mutant count
        for i in range(1, 6):
            moran_process(r=r, N=N, i0=i0, gen_num=i)
        send()
        shutil.rmtree("moran_process_output", ignore_errors=True)

