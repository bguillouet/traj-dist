STUFF = "Hi"

from libc.math cimport fmax
from libc.math cimport fmin


cimport numpy as np
import numpy as np


from basic_euclidean import c_eucl_dist
from basic_geographical import c_great_circle_distance

def c_e_lcss(np.ndarray[np.float64_t,ndim=2] t0, np.ndarray[np.float64_t,ndim=2] t1, eps):
    """
    Usage
    -----
    The Longuest-Common-Subsequence distance between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    eps : float

    Returns
    -------
    lcss : float
           The Longuest-Common-Subsequence distance between trajectory t0 and t1
    """
    cdef int n0,n1,i,j
    cdef np.ndarray[np.float64_t,ndim=2] C
    cdef double x0,y0,x1,y1,lcss

    n0 = len(t0)+1
    n1 = len(t1)+1

    # An (m+1) times (n+1) matrix
    C=np.zeros((n0,n1))

    for i from 1 <= i < n0:
        for j from 1 <= j < n1:
            x0=t0[i-1,0]
            y0=t0[i-1,1]
            x1=t1[j-1,0]
            y1=t1[j-1,1]
            if c_eucl_dist(x0,y0,x1,y1)<eps:
                C[i,j] = C[i-1,j-1] + 1
            else:
                C[i,j] = fmax(C[i,j-1], C[i-1,j])
    lcss = 1-float(C[n0-1,n1-1])/fmin(n0-1,n1-1)
    return lcss



def c_g_lcss(np.ndarray[np.float64_t,ndim=2] t0, np.ndarray[np.float64_t,ndim=2] t1, eps):
    """
    Usage
    -----
    The Longuest-Common-Subsequence distance between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    eps : float

    Returns
    -------
    lcss : float
           The Longuest-Common-Subsequence distance between trajectory t0 and t1
    """
    cdef int n0,n1,i,j
    cdef np.ndarray[np.float64_t,ndim=2] C
    cdef double x0,y0,x1,y1,lcss

    n0 = len(t0)+1
    n1 = len(t1)+1

    # An (m+1) times (n+1) matrix
    C=np.zeros((n0,n1))

    for i from 1 <= i < n0:
        for j from 1 <= j < n1:
            x0=t0[i-1,0]
            y0=t0[i-1,1]
            x1=t1[j-1,0]
            y1=t1[j-1,1]
            if c_great_circle_distance(x0,y0,x1,y1)<eps:
                C[i,j] = C[i-1,j-1] + 1
            else:
                C[i,j] = fmax(C[i,j-1], C[i-1,j])
    lcss = 1-float(C[n0-1,n1-1])/fmin(n0-1,n1-1)
    return lcss
