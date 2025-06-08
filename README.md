# CSC175 (XxYy)
Parallel and distributed Computing
## Main interest of this folder are the three files:
- `mergesort_parallel_mpi_np.py`
- `mergesort_parallel_pool_np.py`
- `mergesort_sequential_np2.py`

### `mergesort_parallel_mpi_np.py`
This uses mpi to perform mergesort. Good for multiple machine, although you can run it on a single machine but the benefits won't be as much.
### `mergesort_parallel_pool_np.py`
This uses pool from multiprocessing to perform mergesort. Good for single machine.
### `mergesort_sequential_np2.py`
This uses npi's sort for the concatenation of right and left as opposed to the manual as seen in `mergesort_sequential_np.py`.
