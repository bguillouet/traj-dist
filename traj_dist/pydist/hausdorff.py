from basic_euclidean import point_to_trajectory, eucl_dist_traj, eucl_dist
from basic_geographical import point_to_path


#############
# euclidean #
#############

def e_directed_hausdorff(t1, t2, mdist, l_t1, l_t2, t2_dist):
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
    dh = max(map(lambda i1: point_to_trajectory(t1[i1], t2, mdist[i1], t2_dist, l_t2), range(l_t1)))
    return dh


def e_hausdorff(t1, t2):
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
    mdist = eucl_dist_traj(t1, t2)
    l_t1 = len(t1)
    l_t2 = len(t2)
    t1_dist = map(lambda it1: eucl_dist(t1[it1], t1[it1 + 1]), range(l_t1 - 1))
    t2_dist = map(lambda it2: eucl_dist(t2[it2], t2[it2 + 1]), range(l_t2 - 1))

    h = max(e_directed_hausdorff(t1, t2, mdist, l_t1, l_t2, t2_dist),
            e_directed_hausdorff(t2, t1, mdist.T, l_t2, l_t1, t1_dist))
    return h


################
# geographical #
################

def g_directed_hausdorff(t1, t2):
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
    n0 = len(t1)
    n1 = len(t2)
    dh = 0
    for j in range(n1):
        dist_j0 = 9e100
        for i in range(n0 - 1):
            dist_j0 = min(dist_j0, point_to_path(t1[i][0], t1[i][1], t1[i + 1][0], t1[i + 1][1], t2[j][0], t2[j][1]))
        dh = max(dh, dist_j0)
    return dh


def g_hausdorff(t1, t2):
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
    h = max(g_directed_hausdorff(t1, t2), g_directed_hausdorff(t2, t1))
    return h
