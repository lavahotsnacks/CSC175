# March 24, 2025 (Monday)

import time
import numpy as np
import csv
import datetime
from multiprocessing import Pool, cpu_count
# from mergesort_sequential_np import merge, mergesort
from mergesort_sequential_np2 import merge, mergesort

def mergeSortParallel(lyst, numproc):
    # Split the list into sublists for each process
    endpoints = np.linspace(0, len(lyst), numproc + 1, dtype=int)
    sublists = [lyst[endpoints[i]:endpoints[i + 1]] for i in range(numproc)]

    # Sort each sublist in parallel
    with Pool(processes=numproc) as pool:
        sorted_sublists = pool.map(mergesort, sublists)

    # Iteratively merge the sorted sublists
    while len(sorted_sublists) > 1:
        args = [(sorted_sublists[i], sorted_sublists[i + 1]) 
                for i in range(0, len(sorted_sublists), 2)]
        # If odd number of sublists, append the last one as is
        if len(sorted_sublists) % 2 == 1:
            args.append((sorted_sublists[-1], np.array([])))
        sorted_sublists = [merge(a, b) for a, b in args]
    return sorted_sublists[0]

def main():
    # Loop over desired number of processes
    for numproc in [2, 4, 8, 16, 32]:
        print(f"numproc = {numproc}")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"mergesort_pool_results_{numproc}_{timestamp}.csv"
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["processes", "N", "seq_time", "par_time"])
            N = 1
            max_N = 10_000_000
            while N <= max_N:
                print(f"N = {N}")
                lystbck = np.random.random(N)

                # Sequential mergesort
                lyst = np.copy(lystbck)
                start = time.time()
                lyst = mergesort(lyst)
                elapsed_seq = time.time() - start
                print(f'Sequential mergesort: {elapsed_seq} sec')

                # Parallel mergesort
                lyst = np.copy(lystbck)
                start = time.time()
                lyst = mergeSortParallel(lyst, numproc)
                elapsed_par = time.time() - start
                print(f'Parallel mergesort: {elapsed_par} sec')

                writer.writerow([numproc, N, elapsed_seq, elapsed_par])
                f.flush()
                N *= 10        # <-- How N increases each iteration

if __name__ == "__main__":
    main()
