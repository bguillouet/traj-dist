from .basic_euclidean import point_to_trajectory, eucl_dist_traj, eucl_dist
from .basic_spherical import point_to_path, great_circle_distance, great_circle_distance_traj


######################
# Euclidean Geometry #
######################

def e_directed_hausdorff(t1, t2, mdist, l_t1, l_t2, t2_dist):
    """
    Usage
    -----
    directed hausdorff distance from trajectory t1 to trajectory t2.

    Parameters
    ----------
    Parameters
    ----------
    param t1 :  l_t1 x 2 numpy_array
    param t2 :  l_t2 x 2 numpy_array
    mdist : len(t1) x len(t2) numpy array, pairwise distance between points of trajectories t1 and t2
    param l_t1: int, length of t1
    param l_t2: int, length of t2
    param t2_dist:  l_t1 x 1 numpy_array,  distances between consecutive points in t2



    Returns
    -------
    dh : float, directed hausdorff from trajectory t1 to trajectory t2
    """
    dh = max([point_to_trajectory(t1[i1], t2, mdist[i1], t2_dist, l_t2) for i1 in range(l_t1)])
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
    t1_dist = [eucl_dist(t1[it1], t1[it1 + 1]) for it1 in range(l_t1 - 1)]
    t2_dist = [eucl_dist(t2[it2], t2[it2 + 1]) for it2 in range(l_t2 - 1)]

    h = max(e_directed_hausdorff(t1, t2, mdist, l_t1, l_t2, t2_dist),
            e_directed_hausdorff(t2, t1, mdist.T, l_t2, l_t1, t1_dist))
    return h


######################
# Spherical Geometry #
######################

def s_directed_hausdorff(lons0, lats0, lons1, lats1, n0, n1, mdist, t0_dist):
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

    dh = 0
    for j in range(n1):
        dist_j0 = 9e100
        for i in range(n0 - 1):
            dist_j0 = min(dist_j0, point_to_path(lons0[i], lats0[i], lons0[i + 1], lats0[i + 1], lons1[j],
                                                 lats1[j], mdist[i, j], mdist[i + 1, j], t0_dist[i]))
        dh = max(dh, dist_j0)
    return dh


def s_hausdorff(t0, t1):
    """
    Usage
    -----
    hausdorff distance between trajectories t1 and t2.

    Parameters
    ----------
    param lons0 :  n0 x 1 numpy_array, longitudes of trajectories t0
    param lats0 :  n0 x 1 numpy_array, lattitudes of trajectories t0
    param lons1 :  n1 x 1 numpy_array, longitudes of trajectories t1
    param lats1 :  n1 x 1 numpy_array, lattitudes of trajectories t1
    param n0: int, length of lons0 and lats0
    param n1: int, length of lons1 and lats1
    mdist : len(t0) x len(t1) numpy array, pairwise distance between points of trajectories t0 and t1
    param t0_dist:  l_t1 x 1 numpy_array,  distances between consecutive points in t0


    Returns
    -------
    h : float, hausdorff from trajectories t1 and t2
    """
    n0 = len(t0)
    n1 = len(t1)
    lats0 = t0[:, 1]
    lons0 = t0[:, 0]
    lats1 = t1[:, 1]
    lons1 = t1[:, 0]

    mdist = great_circle_distance_traj(lons0, lats0, lons1, lats1, n0, n1)

    t0_dist = [great_circle_distance(lons0[it0], lats0[it0], lons0[it0 + 1], lats0[it0 + 1]) for it0 in range(n0 - 1)]
    t1_dist = [great_circle_distance(lons1[it1], lats1[it1], lons1[it1 + 1], lats1[it1 + 1]) for it1 in range(n1 - 1)]

    h = max(s_directed_hausdorff(lons0, lats0, lons1, lats1, n0, n1, mdist, t0_dist),
            s_directed_hausdorff(lons1, lats1, lons0, lats0, n1, n0, mdist.T, t1_dist))
    return h
