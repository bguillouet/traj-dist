import numpy as np
import pandas as pd
import timeit
from scipy.spatial.distance import cdist

data = pd.read_pickle("/Users/bguillouet/These/trajectory_review/data/extracted/starting_from/Caltrain_city_center_v3.pkl")



def eucl_dist(x, y):
    """
    Usage
    -----
    L2-norm between point x and y

    Parameters
    ----------
    param x : numpy_array
    param y : numpy_array

    Returns
    -------
    dist : float
           L2-norm between x and y
    """
    dist = np.linalg.norm(x - y)
    return dist


def point_to_seg(p, s1, s2):
    """
    Usage
    -----
    Point to segment distance between point p and segment delimited by s1 and s2

    Parameters
    ----------
    param p : 1x2 numpy_array
    param s1 : 1x2 numpy_array
    param s2 : 1x2 numpy_array

    Returns
    -------
    dpl: float
         Point to segment distance between p and s
    """
    px = p[0]
    py = p[1]
    p1x = s1[0]
    p1y = s1[1]
    p2x = s2[0]
    p2y = s2[1]
    if p1x == p2x and p1y == p2y:
        dpl = eucl_dist(p, s1)
    else:
        segl = eucl_dist(s1, s2)
        u1 = (((px - p1x) * (p2x - p1x)) + ((py - p1y) * (p2y - p1y)))
        u = u1 / (segl * segl)

        if (u < 0.00001) or (u > 1):
            # // closest point does not fall within the line segment, take the shorter distance
            # // to an endpoint
            ix = eucl_dist(p, s1)
            iy = eucl_dist(p, s2)
            if ix > iy:
                dpl = iy
            else:
                dpl = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = p1x + u * (p2x - p1x)
            iy = p1y + u * (p2y - p1y)
            dpl = eucl_dist(p, np.array([ix, iy]))

    return dpl


def point_to_trajectory(p, t):
    """
    Usage
    -----
    Point-to-trajectory distance between point p and the trajectory t.
    The Point to trajectory distance is the minimum of point-to-segment distance between p and all segment s of t

    Parameters
    ----------
    param p: 1x2 numpy_array
    param t : len(t)x2 numpy_array

    Returns
    -------
    dpt : float,
          Point-to-trajectory distance between p and trajectory t
    """
    dpt = min(map(lambda s1, s2: point_to_seg(p, s1, s2), t[:-1], t[1:]))
    return dpt

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

def eucl_dist_2(t1, t2):
    """
    Usage
    -----
    L2-norm between point x and y

    Parameters
    ----------
    param x : numpy_array
    param y : numpy_array

    Returns
    -------
    dist : float
           L2-norm between x and y
    """
    mdist = cdist(t1, t2, 'euclidean')
    return mdist


def point_to_seg_2(p, s1, s2, dps1, dps2, ds):
    """
    Usage
    -----
    Point to segment distance between point p and segment delimited by s1 and s2

    Parameters
    ----------
    param p : 1x2 numpy_array
    param s1 : 1x2 numpy_array
    param s2 : 1x2 numpy_array

    Returns
    -------
    dpl: float
         Point to segment distance between p and s
    """
    px = p[0]
    py = p[1]
    p1x = s1[0]
    p1y = s1[1]
    p2x = s2[0]
    p2y = s2[1]
    if p1x == p2x and p1y == p2y:
        dpl = dps1
    else:
        segl = ds
        x_diff = p2x - p1x
        y_diff = p2y - p1y
        u1 = (((px - p1x) * x_diff) + ((py - p1y) * y_diff))
        u = u1 / (segl * segl)

        if (u < 0.00001) or (u > 1):
            # closest point does not fall within the line segment, take the shorter distance to an endpoint
            dpl = min(dps1,dps2)
        else:
            # Intersecting point is on the line, use the formula
            ix = p1x + u * x_diff
            iy = p1y + u * y_diff
            dpl = eucl_dist(p, np.array([ix, iy]))

    return dpl

def point_to_trajectory_2(p, t, mdist_p, t_dist, l_t):
    """
    Usage
    -----
    Point-to-trajectory distance between point p and the trajectory t.
    The Point to trajectory distance is the minimum of point-to-segment distance between p and all segment s of t

    Parameters
    ----------
    param p: 1x2 numpy_array
    param t : len(t)x2 numpy_array

    Returns
    -------
    dpt : float,
          Point-to-trajectory distance between p and trajectory t
    """
    dpt = min(map(lambda it: point_to_seg_2(p, t[it], t[it+1], mdist_p[it], mdist_p[it+1], t_dist[it]), range(l_t-1)))
    return dpt


def e_spd_2(t1, t2, mdist, l_t1, l_t2, t2_dist):
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

    spd=sum(map(lambda i1 : point_to_trajectory_2(t1[i1],t2, mdist[i1], t2_dist, l_t2),range(l_t1)))/l_t1
    return spd


def e_sspd_2 (t1, t2):
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
    mdist = eucl_dist_2(t1, t2)
    l_t1 = len(t1)
    l_t2 = len(t2)
    t1_dist = map(lambda it1 : eucl_dist(t1[it1], t1[it1+1]), range(l_t1-1))
    t2_dist = map(lambda it2 : eucl_dist(t2[it2], t2[it2+1]), range(l_t2-1))

    sspd=(e_spd_2(t1, t2, mdist, l_t1, l_t2, t2_dist) + e_spd_2(t2, t1, mdist.T, l_t2, l_t1, t1_dist))/2
    return sspd


traj_list = [group[["lons","lats"]].values for _,group in data.groupby("id_traj")][:100]
nb_traj = len(traj_list)
print(nb_traj)


def plop(func):
    im=0
    M = np.zeros(sum(range(nb_traj)))
    for i in range(nb_traj):
        traj_list_i = traj_list[i]
        for j in range(i + 1, nb_traj):
            traj_list_j = traj_list[j]
            M[im] = func(traj_list_i, traj_list_j)
            im += 1

print(timeit.timeit(lambda: plop(e_sspd), number=1))
print(timeit.timeit(lambda: plop(e_sspd_2), number=1))

traj_1 = data[data.id_traj==1][["lons","lats"]].values
traj_2 = data[data.id_traj == 2][["lons", "lats"]].values


print(timeit.timeit(lambda:e_sspd(traj_1, traj_2), number=100))
print(timeit.timeit(lambda:e_sspd_2(traj_1, traj_2), number=100))

