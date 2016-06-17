from basic_euclidean import eucl_dist
from basic_geographical import great_circle_distance

#############
# euclidean #
#############

def e_edr(t0, t1,eps):
    """
    Usage
    -----
    The Edit Distance on Real sequence between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    eps : float

    Returns
    -------
    edr : float
           The Longuest-Common-Subsequence distance between trajectory t0 and t1
    """
    n0 = len(t0)
    n1 = len(t1)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n1+1) for _ in range(n0+1)]
    for i in range(1, n0+1):
        for j in range(1, n1+1):
            if eucl_dist(t0[i-1],t1[j-1])<eps:
                subcost = 0
            else:
                subcost = 1
            C[i][j] = min(C[i][j-1]+1, C[i-1][j]+1,C[i-1][j-1]+subcost)
    edr = float(C[n0][n1])/max([n0,n1])
    return edr

################
# geographical #
################

def g_edr(t0, t1,eps):
    """
    Usage
    -----
    The Edit Distance on Real sequence between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    eps : float

    Returns
    -------
    edr : float
           The Longuest-Common-Subsequence distance between trajectory t0 and t1
    """
    n0 = len(t0)
    n1 = len(t1)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n1+1) for _ in range(n0+1)]
    for i in range(1, n0+1):
        for j in range(1, n1+1):
            if great_circle_distance(t0[i-1][0],t0[i-1][1],t1[j-1][0],t1[j-1][1])<eps:
                subcost = 0
            else:
                subcost = 1
            C[i][j] = min(C[i][j-1]+1, C[i-1][j]+1,C[i-1][j-1]+subcost)
    edr = float(C[n0][n1])/max([n0,n1])
    return edr
