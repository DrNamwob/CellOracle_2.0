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




def colDeltaCorpartial_numpy(e: np.ndarray, d: np.ndarray, ixs: np.ndarray) -> np.ndarray:
    e = np.asarray(e, dtype=np.float64, order="C")
    d = np.asarray(d, dtype=np.float64, order="C")
    ixs = np.asarray(ixs, dtype=np.int64, order="C")

    rows, cols = e.shape
    nrndm = ixs.shape[1]
    rm = np.zeros((cols, cols), dtype=np.float64)

    eps = 1e-12
    for c in range(cols):
        idx = ixs[c]                   # (nrndm,)
        A = e[:, idx] - e[:, [c]]      # (rows, nrndm)
        A -= A.mean(axis=0, keepdims=True)

        b = d[:, c].astype(np.float64)
        b -= b.mean()
        b_norm = np.linalg.norm(b) + eps

        A_norms = np.linalg.norm(A, axis=0) + eps
        rm[c, idx] = (b @ A) / (b_norm * A_norms)

    return rm
