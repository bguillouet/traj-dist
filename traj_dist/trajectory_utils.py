import numpy as np


def remove_duplicate_adjacent_pts(t, atol=1e-6):
    """
    remove consequent points that are equal (with distance 0 between them)
    consider 2 numbers as equal, if the difference between them < atol (absolute tolerance)
    each degree of latitude is about 111,111 meters. default atol=1e-6 is 1e-1 centimeters' precision

    Parameters
    ----------
    @param t: trajectory points
    @param atol: absolute tolerance.
    @return: t after removing points that are equal

    """
    # np.isclose(a, b): absolute(a - b) <= (atol + rtol * absolute(b))
    # the relative tolerance is negligible in small numbers, hence not used
    diff1 = np.diff(t[:, 0])
    diff1 = np.where(np.isclose(0, diff1, atol=atol, rtol=0), 0, diff1)
    arr1 = np.nonzero(diff1)
    diff2 = np.diff(t[:, 1])
    diff2 = np.where(np.isclose(0, diff2, atol=atol, rtol=0), 0, diff2)
    arr2 = np.nonzero(diff2)
    # union of the 2 arrays
    t1_indices = np.union1d(arr1, arr2)
    t1_indices = np.append(t1_indices, len(t) - 1)
    t = t[t1_indices]
    return t


def increasing(arr):
    return non_decreasing(arr) and arr[0] < arr[-1]


def non_decreasing(arr):
    return all(x <= y for x, y in zip(arr[:-1], arr[1:]))
