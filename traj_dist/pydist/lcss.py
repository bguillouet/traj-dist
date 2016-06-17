from basic_euclidean import eucl_dist
from basic_geographical import great_circle_distance

#############
# euclidean #
#############

def e_lcss(t0, t1,eps):
    """
    Usage
    -----
    The Longuest-Common-Subsequence distance between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    eps : float

    Returns
    -------
    lcss : float
           The Longuest-Common-Subsequence distance between trajectory t0 and t1
    """
    n0 = len(t0)
    n1 = len(t1)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n1+1) for _ in range(n0+1)]
    for i in range(1, n0+1):
        for j in range(1, n1+1):
            if eucl_dist(t0[i-1],t1[j-1])<eps:
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = max(C[i][j-1], C[i-1][j])
    lcss = 1-float(C[n0][n1])/min([n0,n1])
    return lcss

################
# geographical #
################

def g_lcss(t0, t1,eps):
    """
    Usage
    -----
    The Longuest-Common-Subsequence distance between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    eps : float

    Returns
    -------
    lcss : float
           The Longuest-Common-Subsequence distance between trajectory t0 and t1
    """
    n0 = len(t0)
    n1 = len(t1)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n1+1) for _ in range(n0+1)]
    for i in range(1, n0+1):
        for j in range(1, n1+1):
            if great_circle_distance(t0[i-1,0],t0[i-1,1],t1[j-1,0],t1[j-1,1])<eps:
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = max(C[i][j-1], C[i-1][j])
    lcss = 1-float(C[n0][n1])/min([n0,n1])
    return lcss

