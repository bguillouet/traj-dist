import numpy as np
from .basic_euclidean import eucl_dist, eucl_dist_traj
from .basic_spherical import great_circle_distance


######################
# Euclidean Geometry #
######################

def e_erp(t0, t1, g):
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

    n0 = len(t0)
    n1 = len(t1)
    C = np.zeros((n0 + 1, n1 + 1))

    gt0_dist = [abs(eucl_dist(g, x)) for x in t0]
    gt1_dist = [abs(eucl_dist(g, x)) for x in t1]
    mdist = eucl_dist_traj(t0, t1)

    C[1:, 0] = sum(gt0_dist)
    C[0, 1:] = sum(gt1_dist)
    for i in np.arange(n0) + 1:
        for j in np.arange(n1) + 1:
            derp0 = C[i - 1, j] + gt0_dist[i-1]
            derp1 = C[i, j - 1] + gt1_dist[j-1]
            derp01 = C[i - 1, j - 1] + mdist[i-1, j-1]
            C[i, j] = min(derp0, derp1, derp01)
    erp = C[n0, n1]
    return erp


######################
# Spherical Geometry #
######################

def s_erp(t0, t1, g):
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
    n0 = len(t0)
    n1 = len(t1)
    C = np.zeros((n0 + 1, n1 + 1))

    C[1:, 0] = sum([abs(great_circle_distance(g[0], g[1], x[0], x[1])) for x in t0])
    C[0, 1:] = sum([abs(great_circle_distance(g[0], g[1], y[0], y[1])) for y in t1])
    for i in np.arange(n0) + 1:
        for j in np.arange(n1) + 1:
            derp0 = C[i - 1, j] + great_circle_distance(t0[i - 1][0], t0[i - 1][1], g[0], g[1])
            derp1 = C[i, j - 1] + great_circle_distance(g[0], g[1], t1[j - 1][0], t1[j - 1][1])
            derp01 = C[i - 1, j - 1] + great_circle_distance(t0[i - 1][0], t0[i - 1][1], t1[j - 1][0], t1[j - 1][1])
            C[i, j] = min(derp0, derp1, derp01)
    erp = C[n0, n1]
    return erp
