STUFF = "Hi"

from libc.math cimport fmin

cimport numpy as np
import numpy as np
from numpy.math cimport INFINITY

from basic_euclidean import c_eucl_dist
from basic_geographical import c_great_circle_distance

###############
#### DTW ######
###############

def c_e_dtw(np.ndarray[np.float64_t,ndim=2] t0, np.ndarray[np.float64_t,ndim=2] t1):
    """
    Usage
    -----
    The Dynamic-Time Warping distance between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array

    Returns
    -------
    dtw : float
          The Dynamic-Time Warping distance between trajectory t0 and t1
    """
    cdef int n0,n1,i,j
    cdef np.ndarray[np.float64_t,ndim=2] C
    cdef double x0,y0,x1,y1,dtw

    n0 = len(t0)+1
    n1 = len(t1)+1
    C=np.zeros((n0,n1))

    C[1:,0]=INFINITY
    C[0,1:]=INFINITY
    for i from 1 <= i < n0:
        for j from 1 <= j < n1:
            x0=t0[i-1,0]
            y0=t0[i-1,1]
            x1=t1[j-1,0]
            y1=t1[j-1,1]
            C[i,j]=c_eucl_dist(x0,y0,x1,y1) + fmin(fmin(C[i,j-1],C[i-1,j-1]),C[i-1,j])
    dtw = C[n0-1,n1-1]
    return dtw


def c_g_dtw(np.ndarray[np.float64_t,ndim=2] t0, np.ndarray[np.float64_t,ndim=2] t1):
    """
    Usage
    -----
    The Dynamic-Time Warping distance between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array

    Returns
    -------
    dtw : float
          The Dynamic-Time Warping distance between trajectory t0 and t1
    """
    cdef int n0,n1,i,j
    cdef np.ndarray[np.float64_t,ndim=2] C
    cdef double x0,y0,x1,y1,dtw

    n0 = len(t0)+1
    n1 = len(t1)+1
    C=np.zeros((n0,n1))

    C[1:,0]=INFINITY
    C[0,1:]=INFINITY
    for i from 1 <= i < n0:
        for j from 1 <= j < n1:
            x0=t0[i-1,0]
            y0=t0[i-1,1]
            x1=t1[j-1,0]
            y1=t1[j-1,1]
            C[i,j]=c_great_circle_distance(x0,y0,x1,y1) + fmin(fmin(C[i,j-1],C[i-1,j-1]),C[i-1,j])
    dtw = C[n0-1,n1-1]
    return dtw
