import csv
import os
import utils
import threading

#run params
if __name__ == "__main__":
    threads = []
    for i in range(5):
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
