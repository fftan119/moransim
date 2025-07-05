import csv
import os
import random

def moran_process(r, N, i0, gen_num):
    """
    Moran process with fitness-proportional reproduction and uniform death.
    Now outputs birth-death events as [birth_index][type]:[death_index][type].
    """
    assert 0 < i0 < N, "Initial mutant count must be between 1 and N-1"

    i = i0
    population = ['A'] * i + ['B'] * (N - i)

    event_data = []  # Stores tuples: (generation, birth-death string, A count, B count)

    # Output directory
    output_dir = r"[drive num]:\[dir]"
    os.makedirs(output_dir, exist_ok=True)

    # Handle auto-increment filename
    base_name = f"Generation {gen_num}"
    file_path = os.path.join(output_dir, f"{base_name}.csv")
    suffix = 1
    while os.path.exists(file_path):
        file_path = os.path.join(output_dir, f"{base_name}_{suffix}.csv")
        suffix += 1

    gen = 0
    while 0 < i < N:
        # Birth selection (fitness-proportional)
        weights = [r if ind == 'A' else 1 for ind in population]
        birth_index = random.choices(range(N), weights=weights)[0]
        birth_type = population[birth_index]

        # Death selection (uniform)
        death_index = random.randrange(N)
        death_type = population[death_index]

        # Replace the individual
        population[death_index] = birth_type

        # Count new A/B
        i = population.count('A')
        b = N - i

        # Format event string
        event = f"{birth_index}{birth_type}:{death_index}{death_type}"
        event_data.append((gen, event, i, b))
        gen += 1

    # Write to CSV
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Generation', 'Event', 'A', 'B'])
        writer.writerows(event_data)

    print(f"Simulation complete. Saved to: {file_path}")


# Example usage
if __name__ == "__main__":
    moran_process(r=1.2, N=20, i0=3, gen_num=1)
