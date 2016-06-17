STUFF = "Hi"

from libc.math cimport fmin,fabs

cimport numpy as np
import numpy as np
from numpy.math cimport INFINITY

from basic_euclidean import c_eucl_dist
from basic_geographical import c_great_circle_distance

###############
#### ERP ######
###############

def c_e_erp(np.ndarray[np.float64_t,ndim=2] t0, np.ndarray[np.float64_t,ndim=2] t1, np.ndarray[np.float64_t,ndim=1] g):
    """
    Usage
    -----
    The Edit distance with Real Penalty between trajectory t0 and t1.

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
    cdef double x0,y0,x1,y1,dtw,gx,gy,edgei,edgej

    gx=g[0]
    gy=g[1]

    n0 = len(t0)+1
    n1 = len(t1)+1
    C=np.zeros((n0,n1))
    edgei=0
    for i from 1 <= i < n0:
        x0=t0[i-1,0]
        y0=t0[i-1,1]
        edgei += fabs(c_eucl_dist(x0,y0,gx,gy))
    C[1:,0]=edgei

    edgej=0
    for j from 1 <= j < n1:
        x1=t1[j-1,0]
        y1=t1[j-1,1]
        edgej += fabs(c_eucl_dist(x1,y1,gx,gy))
    C[0,1:]=edgej

    for i from 1 <= i < n0:
        for j from 1 <= j < n1:
            x0=t0[i-1,0]
            y0=t0[i-1,1]
            x1=t1[j-1,0]
            y1=t1[j-1,1]
            derp0 = C[i-1,j] + c_eucl_dist(x0,y0,gx,gy)
            derp1 = C[i,j-1] + c_eucl_dist(gx,gy,x1,y1)
            derp01 = C[i-1,j-1] + c_eucl_dist(x0,y0,x1,y1)
            C[i,j] = fmin(derp0,fmin(derp1,derp01))
    erp = C[n0-1,n1-1]
    return erp


def c_g_erp(np.ndarray[np.float64_t,ndim=2] t0, np.ndarray[np.float64_t,ndim=2] t1, np.ndarray[np.float64_t,ndim=1] g):
    """
    Usage
    -----
    The Edit distance with Real Penalty between trajectory t0 and t1.

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
    cdef double x0,y0,x1,y1,dtw,gx,gy,edgei,edgej

    gx=g[0]
    gy=g[1]

    n0 = len(t0)+1
    n1 = len(t1)+1
    C=np.zeros((n0,n1))

    edgei=0.0
    for i from 1 <= i < n0:
        x0=t0[i-1,0]
        y0=t0[i-1,1]
        edgei += fabs(c_great_circle_distance(x0,y0,gx,gy))
    C[1:,0]=edgei

    edgej=0.0
    for j from 1 <= j < n1:
        x1=t1[j-1,0]
        y1=t1[j-1,1]
        edgej += fabs(c_great_circle_distance(x1,y1,gx,gy))
    C[0,1:]=edgej
    for i from 1 <= i < n0:
        for j from 1 <= j < n1:
            x0=t0[i-1,0]
            y0=t0[i-1,1]
            x1=t1[j-1,0]
            y1=t1[j-1,1]
            derp0 = C[i-1,j] + c_great_circle_distance(x0,y0,gx,gy)
            derp1 = C[i,j-1] + c_great_circle_distance(gx,gy,x1,y1)
            derp01 = C[i-1,j-1] + c_great_circle_distance(x0,y0,x1,y1)
            C[i,j] = fmin(derp0,fmin(derp1,derp01))
    erp = C[n0-1,n1-1]
    return erp
