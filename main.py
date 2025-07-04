import csv
import os
import random

def moran_process(B, P, M, gen_num):
    #definitions
    A = P - B
    population = ['A'] * A + ['B'] * B
    generation_data = []

    #dir setup
    output_dir = r"[driveLetter]:\dir"
    os.makedirs(output_dir, exist_ok=True)

    #read base directory for existing files with same gen num, increment by 1 if it exists. adds _[run num] if none exists.
    base_name = f"Generation {gen_num}"
    file_path = os.path.join(output_dir, f"{base_name}.csv")
    i = 1
    while os.path.exists(file_path):
        file_path = os.path.join(output_dir, f"{base_name}_{i}.csv")
        i += 1

    #record initial state
    generation_data.append(f"A={population.count('A')} B={population.count('B')}")

    #run Moran process until fixation (all mutant A, or B)
    while 0 < population.count('B') < P:
        #1. select a reproducer
        reproducer = random.choice(population)
        #2. apply mutation
        if random.random() < M:
            offspring = 'B' if reproducer == 'A' else 'A'
        else:
            offspring = reproducer
        # 3. select someone to be replaced
        replaced_index = random.randint(0, P - 1)
        population[replaced_index] = offspring
        # 4. record new generation
        generation_data.append(f"A={population.count('A')} B={population.count('B')}")

    #write to csv
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for line in generation_data:
            writer.writerow([line])

    print(f"Simulation complete. Saved to: {file_path}")

#run params
if __name__ == "__main__":
    moran_process(B=3, P=20, M=0.01, gen_num=1)
