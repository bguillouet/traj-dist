from basic_euclidean import point_to_trajectory
from basic_geographical import point_to_path

#############
# euclidean #
#############

def e_directed_hausdorff (t1, t2):
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
    dh=max(map(lambda p : point_to_trajectory(p,t2),t1))
    return dh


def e_hausdorff(t1,t2):
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
    h=max(e_directed_hausdorff(t1,t2), e_directed_hausdorff(t2,t1))
    return h

################
# geographical #
################

def g_directed_hausdorff (t1, t2):
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
    n0=len(t1)
    n1=len(t2)
    dh=0
    for j in range(n1) :
        dist_j0=9e100
        for i in range(n0-1):
            dist_j0=min(dist_j0,point_to_path(t1[i][0],t1[i][1],t1[i+1][0],t1[i+1][1],t2[j][0],t2[j][1]))
        dh=max(dh,dist_j0)
    return dh


def g_hausdorff(t1,t2):
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
    h=max(g_directed_hausdorff(t1,t2), g_directed_hausdorff(t2,t1))
    return h

