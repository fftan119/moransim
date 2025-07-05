import random
import threading
def moran(B, P, M, max_generations=10000000):
    A = P - B
    population = ['A'] * A + ['B'] * B
    generation_data = []
    #record initial state
    generation_data.append(f"A={population.count('A')} B={population.count('B')}")

    generation_count = 0
    #run Moran process until fixation (all mutant A, or B) or max generations reached
    while 0 < population.count('B') < P and generation_count < max_generations:
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
        generation_count += 1

    # Add timeout indicator if max generations reached
    if generation_count >= max_generations:
        generation_data.append("TIMEOUT: Max generations reached")
    
    return generation_data

def generate_initial_conditions():
    P = random.randint(1,20)
    B = random.randint(1, P - 1)
    return B, P



_write_lock = threading.Lock()

def run_simulation(M):
    B, P = generate_initial_conditions()
    print(f"Running simulation with B={B}, P={P}, M={M}")
    generations = moran(B, P, M)
    final_state = generations[-1]
    
    # Handle timeout case
    if "TIMEOUT" in final_state:
        result = "TIMEOUT: No fixation reached\n"
        print(f"Simulation timed out: B={B}, P={P}, M={M}")
    else:
        result = "B dominated\n" if "A=0" in final_state else "A dominated\n"
        print(f"Simulation completed: B={B}, P={P}, M={M}, Result: {result.strip()}")
    
    with _write_lock:
        with open("dominance_result.txt", "a") as f:
            f.write(result)
