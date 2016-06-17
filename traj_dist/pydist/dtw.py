import numpy as np
from basic_euclidean import eucl_dist
from basic_geographical import great_circle_distance

#############
# euclidean #
#############

def e_dtw(t0,t1):
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

    n0 = len(t0)
    n1 = len(t1)
    C=np.zeros((n0+1,n1+1))
    C[1:,0]=float('inf')
    C[0,1:]=float('inf')
    for i in np.arange(n0)+1:
        for j in np.arange(n1)+1:
            C[i,j]=eucl_dist(t0[i-1],t1[j-1]) + min(C[i,j-1],C[i-1,j-1],C[i-1,j])
    dtw = C[n0,n1]
    return dtw

################
# geographical #
################

def g_dtw(t0,t1):
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
    n0 = len(t0)
    n1 = len(t1)
    C=np.zeros((n0+1,n1+1))
    C[1:,0]=float('inf')
    C[0,1:]=float('inf')
    for i in np.arange(n0)+1:
        for j in np.arange(n1)+1:
            C[i,j]=great_circle_distance(t0[i-1][0],t0[i-1][1],t1[j-1][0],t1[j-1][1]) + min(C[i,j-1],C[i-1,j-1],C[i-1,j])
    dtw = C[n0,n1]
    return dtw