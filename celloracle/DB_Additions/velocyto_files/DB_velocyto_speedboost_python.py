# This code is the Python only adaptation of velocyto's speedboost modules that originally used Cython.
# The code has been modified to remove Cython dependencies and make it compatible with pure Python execution

import numpy as np

def colDeltaCor_numpy(e: np.ndarray, d: np.ndarray, threads=None) -> np.ndarray:
    """
    e: (rows, cols) float64
    d: (rows, cols) float64
    returns rm: (cols, cols) float64 where rm[c, i] is corr(b_c, e[:,i]-e[:,c])
    """
    e = np.asarray(e, dtype=np.float64, order="C")
    d = np.asarray(d, dtype=np.float64, order="C")
    rows, cols = e.shape
    rm = np.zeros((cols, cols), dtype=np.float64)

    eps = 1e-12
    for c in range(cols):
        A = e - e[:, [c]]              # (rows, cols)
        A -= A.mean(axis=0, keepdims=True)

        b = d[:, c].astype(np.float64)
        b -= b.mean()
        b_norm = np.linalg.norm(b) + eps

        # norms of each column of A
        A_norms = np.linalg.norm(A, axis=0) + eps

        # dot products b^T A for all columns at once
        rm[c, :] = (b @ A) / (b_norm * A_norms)

    return rm



# THIS ONE DOES NOT HAVE PARALLELIZATION
# def colDeltaCorpartial_numpy(
#     e: np.ndarray,
#     d: np.ndarray,
#     ixs: np.ndarray,
#     threads: int | None = None,   # <-- added
# ) -> np.ndarray:
#     e = np.asarray(e, dtype=np.float64, order="C")
#     d = np.asarray(d, dtype=np.float64, order="C")
#     ixs = np.asarray(ixs, dtype=np.int64, order="C")

#     rows, cols = e.shape
#     nrndm = ixs.shape[1]
#     rm = np.zeros((cols, cols), dtype=np.float64)

#     eps = 1e-12
#     for c in range(cols):
#         idx = ixs[c]                   # (nrndm,)
#         A = e[:, idx] - e[:, [c]]      # (rows, nrndm)
#         A -= A.mean(axis=0, keepdims=True)

#         b = d[:, c]
#         b = b - b.mean()
#         b_norm = np.linalg.norm(b) + eps

#         A_norms = np.linalg.norm(A, axis=0) + eps
#         rm[c, idx] = (b @ A) / (b_norm * A_norms)

#     return rm


# THIS ONE HAS PARALLELIZATION ADDED!!!!!!
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np

def colDeltaCorpartial_numpy(
    e: np.ndarray,
    d: np.ndarray,
    ixs: np.ndarray,
    threads: int | None = None,
) -> np.ndarray:
    
    print('Using parallelized colDeltaCorpartial_numpy function with', threads, 'threads.')
    e = np.asarray(e, dtype=np.float64, order="C")
    d = np.asarray(d, dtype=np.float64, order="C")
    ixs = np.asarray(ixs, dtype=np.int64, order="C")

    rows, cols = e.shape
    rm = np.zeros((cols, cols), dtype=np.float64)
    eps = 1e-12

    def _one_col(c: int):
        idx = ixs[c]
        A = e[:, idx] - e[:, [c]]
        A -= A.mean(axis=0, keepdims=True)

        b = d[:, c]
        b = b - b.mean()
        b_norm = np.linalg.norm(b) + eps

        A_norms = np.linalg.norm(A, axis=0) + eps
        vals = (b @ A) / (b_norm * A_norms)
        return c, idx, vals

    n_workers = threads if (threads is not None and threads > 0) else (os.cpu_count() // 2 or 1)

    if n_workers <= 1:
        for c in range(cols):
            c, idx, vals = _one_col(c)
            rm[c, idx] = vals
        return rm

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for c, idx, vals in ex.map(_one_col, range(cols)):
            rm[c, idx] = vals

    return rm

