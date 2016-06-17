STUFF = "Hi"

#cython: boundscheck=False
#cython: wraparound=False

from libc.math cimport fmin

from basic_euclidean import c_point_to_trajectory
from basic_geographical import c_point_to_path
cimport numpy as np
from numpy.math cimport INFINITY



################
## Euclidean ###
################

cdef double _e_spd (np.ndarray[np.float64_t,ndim=2] t1, np.ndarray[np.float64_t,ndim=2] t2):
    """
    Usage
    -----
    The spd-distance of trajectory t2 from trajectory t1
    The spd-distance is the sum of the all the point-to-trajectory distance of points of t1 from trajectory t2

    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array
    param t2 :  len(t2)x2 numpy_array

    Returns
    -------
    spd : float
           spd-distance of trajectory t2 from trajectory t1
    """
    cdef int nt,i
    cdef double spd,px,py

    nt=len(t1)
    spd=0
    for i from 0 <= i < (nt):
        px=t1[i,0]
        py=t1[i,1]
        spd=spd+c_point_to_trajectory(px,py,t2)
    spd=spd/nt
    return spd

def c_e_spd (np.ndarray[np.float64_t,ndim=2] t1, np.ndarray[np.float64_t,ndim=2] t2):
    """
    Usage
    -----
    The spd-distance of trajectory t2 from trajectory t1
    The spd-distance is the sum of the all the point-to-trajectory distance of points of t1 from trajectory t2

    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array
    param t2 :  len(t2)x2 numpy_array

    Returns
    -------
    spd : float
           spd-distance of trajectory t2 from trajectory t1
    """
    cdef int nt,i
    cdef double spd,px,py

    nt=len(t1)
    spd=0
    for i from 0 <= i < (nt):
        px=t1[i,0]
        py=t1[i,1]
        spd=spd+c_point_to_trajectory(px,py,t2)
    spd=spd/nt
    return spd

def c_e_sspd (np.ndarray[np.float64_t,ndim=2] t1, np.ndarray[np.float64_t,ndim=2] t2):
    """
    Usage
    -----
    The sspd-distance between trajectories t1 and t2.
    The sspd-distance is the mean of the spd-distance between of t1 from t2 and the spd-distance of t2 from t1.

    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array
    param t2 :  len(t2)x2 numpy_array

    Returns
    -------
    sspd : float
            sspd-distance of trajectory t2 from trajectory t1
    """
    cdef double sspd
    sspd=(_e_spd(t1,t2) + _e_spd(t2,t1))/2
    return sspd


###################
## Geographical ###
###################

cdef double _g_spd(np.ndarray[np.float64_t,ndim=2] t1, np.ndarray[np.float64_t,ndim=2] t2):
    """
    Usage
    -----
    The spd-distance of trajectory t2 from trajectory t1
    The spd-distance is the sum of the all the point-to-path distance of points of t1 from trajectory t2

    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array
    param t2 :  len(t2)x2 numpy_array

    Returns
    -------
    spd : float
           spd-distance of trajectory t2 from trajectory t1
    """
    cdef int n1,i,j
    cdef double dist,dist_j0
    cdef np.ndarray[np.float64_t,ndim=1] lats0,lons0,lats1,lons1

    n0=len(t1)
    n1=len(t2)
    lats0=t1[:,1]
    lons0=t1[:,0]
    lats1=t2[:,1]
    lons1=t2[:,0]
    dist=0
    for j from 0 <= j < n1  :
        dist_j0=INFINITY
        for i from 0 <= i < (n0-1):
            dist_j0=fmin(dist_j0,c_point_to_path(lons0[i],lats0[i],lons0[i+1],lats0[i+1],lons1[j],lats1[j]))
        dist=dist+dist_j0
    dist=float(dist)/n1
    return dist


def c_g_spd(np.ndarray[np.float64_t,ndim=2] t1, np.ndarray[np.float64_t,ndim=2] t2):
    """
    Usage
    -----
    The spd-distance of trajectory t2 from trajectory t1
    The spd-distance is the sum of the all the point-to-path distance of points of t1 from trajectory t2

    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array
    param t2 :  len(t2)x2 numpy_array

    Returns
    -------
    spd : float
           spd-distance of trajectory t2 from trajectory t1
    """
    cdef int n1,i,j
    cdef double dist,dist_j0
    cdef np.ndarray[np.float64_t,ndim=1] lats0,lons0,lats1,lons1

    n0=len(t1)
    n1=len(t2)
    lats0=t1[:,1]
    lons0=t1[:,0]
    lats1=t2[:,1]
    lons1=t2[:,0]
    dist=0
    for j from 0 <= j < n1:
        dist_j0=INFINITY
        for i from 0 <= i < (n0-1):
            dist_j0=fmin(dist_j0,c_point_to_path(lons0[i],lats0[i],lons0[i+1],lats0[i+1],lons1[j],lats1[j]))
        dist=dist+dist_j0
    dist=float(dist)/n1
    return dist


cdef double _g_sspd(np.ndarray[np.float64_t,ndim=2] t1, np.ndarray[np.float64_t,ndim=2] t2):
    """
    Usage
    -----
    The sspd-distance between trajectories t1 and t2.
    The sspd-distance is the mean of the spd-distance between of t1 from t2 and the spd-distance of t2 from t1.

    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array
    param t2 :  len(t2)x2 numpy_array


    Returns
    -------
    sspd : float
            sspd-distance of trajectory t2 from trajectory t1
    """

    cdef double dist

    dist=_g_spd(t1,t2) + _g_spd(t2,t1)
    return dist

def c_g_sspd(np.ndarray[np.float64_t,ndim=2] t1, np.ndarray[np.float64_t,ndim=2] t2):
    """
    Usage
    -----
    The sspd-distance between trajectories t1 and t2.
    The sspd-distance is the mean of the spd-distance between of t1 from t2 and the spd-distance of t2 from t1.

    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array
    param t2 :  len(t2)x2 numpy_array

    Returns
    -------
    sspd : float
            sspd-distance of trajectory t2 from trajectory t1
    """
    dist=_g_spd(t1,t2) + _g_spd(t2,t1)
    return dist
