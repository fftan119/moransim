import csv
import os
import utils
import threading

#run params
if __name__ == "__main__":
    threads = []
    for i in range(500):
        t = threading.Thread(target=utils.run_simulation, kwargs={'M': 0.01}, name=f"SimThread-{i}")
        t.start()
        threads.append(t)
        print(f"Started thread {i}")
    
    print("Waiting for all threads to complete...")
    for i, t in enumerate(threads):
        print(f"Joining thread {i}...")
        t.join()
        print(f"Thread {i} completed")
    
    print("All simulations completed!")
    dominance_file = "dominance_result.txt"
    a_dominates = 0
    b_dominates = 0
    total = 0

    with open(dominance_file, "r") as f:
        for line in f:
            line = line.strip()
            if line == "A dominated":
                a_dominates += 1
                total += 1
            elif line == "B dominated":
                b_dominates += 1
                total += 1

    if total > 0:
        print(f"A dominated in {a_dominates/total*100:.2f}% of runs.")
        print(f"B dominated in {b_dominates/total*100:.2f}% of runs.")
    else:
        print("No results found in dominance_results.txt.")