# March 24, 2025 (Monday)

'''
Topics in Parallel and Distributed Computing:
Introducing Concurrency in Undergraduate Courses
(Edited by Prasad, Gupta, Rosenberg, Sussman, Weems)
Page 56

Figure 3.27 
Parallel mergesort using Pool/map.
The code relies on sequential mergesort from Figure 3.17.
'''

from mpi4py import MPI
import time, random, sys
import numpy as np
import csv
import datetime
# from mergesort_sequential_np import merge, mergesort
from mergesort_sequential_np2 import merge, mergesort

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()  # <--- Number of processes, set by mpiexec/mpirun

    if rank == 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"mergesort_results_{timestamp}.csv"
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["processes", "N", "seq_time", "par_time"])
            N = 1
            max_N = 10_000_000
            while N <= max_N:
                print(f"N = {N}")
                lystbck = np.random.random(N)  # Use NumPy to generate random numbers.

                # Sequential mergesort a copy of the list.
                lyst = np.copy(lystbck)
                start = time.time()
                lyst = mergesort(lyst)
                elapsed_seq = time.time() - start
                print(f'Sequential mergesort: {elapsed_seq} sec')

                # Parallel mergesort.
                lyst = np.copy(lystbck)
                start = time.time()
                lyst = mergeSortParallel(lyst, comm, size)
                elapsed_par = time.time() - start
                print(f'Parallel mergesort: {elapsed_par} sec')

                # Write results to CSV
                writer.writerow([size, N, elapsed_seq, elapsed_par])
                f.flush()

                N *= 10
    else:
        # Loop to match the number of N values processed by rank 0
        N = 1
        max_N = 10_000_000
        while N <= max_N:
            mergeSortParallel(None, comm, size)
            N *= 10

def mergeSortParallel(lyst, comm, size):
    rank = comm.Get_rank()

    if rank == 0:
        numproc = size  # <--- Number of processes, set externally
        endpoints = np.linspace(0, len(lyst), numproc + 1, dtype=int)
        sublists = [lyst[endpoints[i]:endpoints[i + 1]] for i in range(numproc)]
    else:
        sublists = None

    # Scatter sublists to all processes.
    sublist = comm.scatter(sublists, root=0)

    # Each process sorts its sublist.
    sorted_sublist = mergesort(sublist)

    # Gather sorted sublists back at root.
    gathered = comm.gather(sorted_sublist, root=0)

    if rank == 0:
        # Iteratively merge the sorted sublists.
        while len(gathered) > 1:
            args = [(gathered[i], gathered[i + 1]) for i in range(0, len(gathered), 2)]
            gathered = [merge(a, b) for a, b in args]
        return gathered[0]
    else:
        return None

if __name__ == "__main__":
    main()
    # To control the number of processes, run this script with:
    # mpiexec -n 4 python mergesort_parallel_pool&map_np.py
    # Replace 4 with 8, 16, etc. as desired.