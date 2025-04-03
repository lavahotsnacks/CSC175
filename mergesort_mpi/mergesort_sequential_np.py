# March 24, 2025 (Monday)

'''
Topics in Parallel and Distributed Computing:
Introducing Concurrency in Undergraduate Courses
(Edited by Prasad, Gupta, Rosenberg, Sussman, Weems)
Page 46
'''

import numpy as np

def merge(left, right):
    ret = np.empty(len(left) + len(right), dtype=left.dtype)
    li = ri = idx = 0
    while li < len(left) and ri < len(right):
        if left[li] < right[ri]:
            ret[idx] = left[li]
            li += 1
        else:
            ret[idx] = right[ri]
            ri += 1
        idx += 1
    if li < len(left):
        ret[idx:] = left[li:]
    else:
        ret[idx:] = right[ri:]
    return ret

def mergesort(lyst):
    if len(lyst) <= 1:
        return lyst
    ind = len(lyst) // 2
    return merge(mergesort(lyst[:ind]), mergesort(lyst[ind:]))