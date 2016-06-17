import numpy as np
from basic_euclidean import point_to_trajectory
from basic_geographical import point_to_path
###############
## euclidean ##
###############

def e_spd (t1, t2):
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
    spd=sum(map(lambda p : point_to_trajectory(p,t2),t1))/len(t1)
    return spd

def e_sspd (t1, t2):
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
    sspd=(e_spd(t1,t2) + e_spd(t2,t1))/2
    return sspd

#################
## geographical##
#################

def g_spd(t1, t2):
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
    n0 = len(t1)
    n1 = len(t2)
    lats0 = t1[:, 1]
    lons0 = t1[:, 0]
    lats1 = t2[:, 1]
    lons1 = t2[:, 0]
    dist = 0
    for j in range(n1):
        dist_j0 = 9e100
        for i in range(n0 - 1):
            dist_j0 = np.min((dist_j0, point_to_path(lons0[i], lats0[i], lons0[i + 1], lats0[i + 1], lons1[j],
                                                     lats1[j])))
        dist = dist + dist_j0
    dist = float(dist) / n1
    return dist


def g_sspd(t1, t2):
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
    dist = g_spd(t1, t2) + g_spd(t2, t1)
    return dist
