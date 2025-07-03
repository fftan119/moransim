import csv
import os
import random

def moran_process(B, P, M, gen_num):
    # Setup
    A = P - B
    population = ['A'] * A + ['B'] * B
    generation_data = []

    # Directory setup
    output_dir = r"D:\MITACS LLM Research Library\Paper 2 - Reversability of Moran Process\Moran Process Sim\generation data"
    os.makedirs(output_dir, exist_ok=True)

    # 🔁 Auto-increment filename if it already exists
    base_name = f"Generation {gen_num}"
    file_path = os.path.join(output_dir, f"{base_name}.csv")
    i = 1
    while os.path.exists(file_path):
        file_path = os.path.join(output_dir, f"{base_name}_{i}.csv")
        i += 1

    # Record initial state
    generation_data.append(f"A={population.count('A')} B={population.count('B')}")

    # Run Moran process until fixation
    while 0 < population.count('B') < P:
        # 1. Select a reproducer
        reproducer = random.choice(population)
        # 2. Apply mutation
        if random.random() < M:
            offspring = 'B' if reproducer == 'A' else 'A'
        else:
            offspring = reproducer
        # 3. Select someone to be replaced
        replaced_index = random.randint(0, P - 1)
        population[replaced_index] = offspring
        # 4. Record new generation
        generation_data.append(f"A={population.count('A')} B={population.count('B')}")

    # Write to CSV
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for line in generation_data:
            writer.writerow([line])

    print(f"Simulation complete. Saved to: {file_path}")

# Example run
if __name__ == "__main__":
    moran_process(B=3, P=20, M=0.01, gen_num=1)