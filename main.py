import csv
import os
import random

def moran_process(r, N, i0, gen_num):
    """
    Moran process with fitness-proportional reproduction and uniform death.
    Arguments:
        r        : fitness of mutant (A)
        N        : total population size
        i0       : initial number of mutants (0 < i0 < N)
        gen_num  : generation number (used for file naming)
    """
    assert 0 < i0 < N, "Initial mutant count must be between 1 and N-1"

    i = i0
    generation_data = [(0, i, N - i)]  # (generation, A, B)

    # Output directory
    output_dir = r"[drive]:\[dir]"
    os.makedirs(output_dir, exist_ok=True)

    # Handle auto-increment filename
    base_name = f"Generation {gen_num}"
    file_path = os.path.join(output_dir, f"{base_name}.csv")
    suffix = 1
    while os.path.exists(file_path):
        file_path = os.path.join(output_dir, f"{base_name}_{suffix}.csv")
        suffix += 1

    gen = 1
    while 0 < i < N:
        # Reproduction
        prob_A_birth = (r * i) / (r * i + (N - i))
        offspring = 'A' if random.random() < prob_A_birth else 'B'

        # Death
        prob_A_death = i / N
        victim = 'A' if random.random() < prob_A_death else 'B'

        # Update population
        if offspring == 'A' and victim == 'B':
            i += 1
        elif offspring == 'B' and victim == 'A':
            i -= 1
        # else: no net change

        generation_data.append((gen, i, N - i))
        gen += 1

    # Write to CSV
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Generation', 'A', 'B'])
        writer.writerows(generation_data)

    print(f"Simulation complete. Saved to: {file_path}")


# Example usage
if __name__ == "__main__":
    moran_process(r=1.2, N=20, i0=3, gen_num=1)
