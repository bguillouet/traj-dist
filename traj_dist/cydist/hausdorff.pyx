STUFF = "Hi"

from libc.math cimport fmax
from libc.math cimport fmin

cimport numpy as np

from basic_euclidean import c_point_to_trajectory
from basic_geographical import c_point_to_path

#############
# Euclidean #
#############

cdef double _e_directed_hausdorff ( np.ndarray[np.float64_t,ndim=2] t1,np.ndarray[np.float64_t,ndim=2] t2):
    """
    Usage
    -----
    directed hausdorff distance from trajectory t1 to trajectory t2.

    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array
    param t2 :  len(t2)x2 numpy_array

    Returns
    -------
    dh : float, directed hausdorff from trajectory t1 to trajectory t2
    """
    cdef double dh
    cdef int n1,i

    dh=0.0
    n1 = len(t1)
    for i from 0 <= i < n1:
        dh=fmax(c_point_to_trajectory(t1[i,0],t1[i,1],t2),dh)
    return dh

def c_e_directed_hausdorff ( np.ndarray[np.float64_t,ndim=2] t1,np.ndarray[np.float64_t,ndim=2] t2):
    """
    Usage
    -----
    directed hausdorff distance from trajectory t1 to trajectory t2.

    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array
    param t2 :  len(t2)x2 numpy_array

    Returns
    -------
    dh : float, directed hausdorff from trajectory t1 to trajectory t2
    """
    cdef double dh
    cdef int n1,i

    dh=0.0
    n1 = len(t1)
    for i from 0 <= i < n1:
        dh=fmax(c_point_to_trajectory(t1[i,0],t1[i,1],t2),dh)
    return dh


def c_e_hausdorff(np.ndarray[np.float64_t,ndim=2] t1,np.ndarray[np.float64_t,ndim=2] t2):
    """
    Usage
    -----
    hausdorff distance between trajectories t1 and t2.

    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array
    param t2 :  len(t2)x2 numpy_array

    Returns
    -------
    h : float, hausdorff from trajectories t1 and t2
    """
    cdef double h

    h=fmax(_e_directed_hausdorff(t1,t2),_e_directed_hausdorff(t2,t1))
    return h


################
# Geographical #
################


cdef double _g_directed_hausdorff ( np.ndarray[np.float64_t,ndim=2] t1,np.ndarray[np.float64_t,ndim=2] t2):
    """
    Usage
    -----
    directed hausdorff distance from trajectory t1 to trajectory t2.

    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array
    param t2 :  len(t2)x2 numpy_array

    Returns
    -------
    dh : float, directed hausdorff from trajectory t1 to trajectory t2
    """
    cdef double dh,dist_k0
    cdef int n0,n1,i,j
    cdef np.ndarray[np.float64_t,ndim=1] lats0,lons0,lats1,lons1

    n1 = len(t2)
    n0 = len(t1)
    lats0=t1[:,1]
    lons0=t1[:,0]
    lats1=t2[:,1]
    lons1=t2[:,0]

    dh=0
    for j from 0 <= j < n1  :
        dist_j0=9e100
        for i from 0 <= i < (n0-1):
            dist_j0=fmin(dist_j0,c_point_to_path(lons0[i],lats0[i],lons0[i+1],lats0[i+1],lons1[j],lats1[j]))
        dh=fmax(dh,dist_j0)
    return dh

def c_g_directed_hausdorff ( np.ndarray[np.float64_t,ndim=2] t1,np.ndarray[np.float64_t,ndim=2] t2):
    """
    Usage
    -----
    directed hausdorff distance from trajectory t1 to trajectory t2.

    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array
    param t2 :  len(t2)x2 numpy_array

    Returns
    -------
    dh : float, directed hausdorff from trajectory t1 to trajectory t2
    """
    cdef double dh,dist_k0
    cdef int n0,n1,i,j
    cdef np.ndarray[np.float64_t,ndim=1] lats0,lons0,lats1,lons1

    n1 = len(t2)
    n0 = len(t1)
    lats0=t1[:,1]
    lons0=t1[:,0]
    lats1=t2[:,1]
    lons1=t2[:,0]

    dh=0
    for j from 0 <= j < n1  :
        dist_j0=9e100
        for i from 0 <= i < (n0-1):
            dist_j0=fmin(dist_j0,c_point_to_path(lons0[i],lats0[i],lons0[i+1],lats0[i+1],lons1[j],lats1[j]))
        dh=fmax(dh,dist_j0)
    return dh

def c_g_hausdorff(np.ndarray[np.float64_t,ndim=2] t1,np.ndarray[np.float64_t,ndim=2] t2):
    """
    Usage
    -----
    hausdorff distance between trajectories t1 and t2.

    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array
    param t2 :  len(t2)x2 numpy_array

    Returns
    -------
    h : float, hausdorff from trajectories t1 and t2
    """
    cdef double h

    h=fmax(_g_directed_hausdorff(t1,t2),_g_directed_hausdorff(t2,t1))
    return h