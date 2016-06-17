# # Computing the Frechet distance between two polygonal curves.
## International Journal of Computational Geometry & Applications.
## Vol 5, Nos 1&2 (1995) 75-91

from basic_euclidean import eucl_dist, point_to_seg, circle_line_intersection


def free_line(p, eps, s):
    """
    Usage
    -----
    Return the free space in the segment s, from point p.
    This free space is the set of all point in s whose distance from p is at most eps.
    Since s is a segment, the free space is also a segment.
    We return a 1x2 array whit the fraction of the segment s which are in the free space.
    If no part of s are in the free space, return [-1,-1]

    Parameters
    ----------
    param p : 1x2 numpy_array, centre of the circle
    param eps : float, radius of the circle
    param s : 2x2 numpy_array, line

    Returns
    -------
    lf : 1x2 numpy_array
         fraction of segment which is in the free space (i.e [0.3,0.7], [0.45,1], ...)
         If no part of s are in the free space, return [-1,-1]
    """
    px = p[0]
    py = p[1]
    s1x = s[0, 0]
    s1y = s[0, 1]
    s2x = s[1, 0]
    s2y = s[1, 1]
    if s1x == s2x and s1y==s2y:
        if eucl_dist(p, s[0]) > eps:
            lf = [-1, -1]
        else:
            lf = [0, 1]
    else:
        if point_to_seg(p, s[0], s[1]) > eps:
            #print("No Intersection")
            lf = [-1, -1]
        else:
            segl = eucl_dist(s[0], s[1])
            segl2 = segl * segl
            intersect = circle_line_intersection(px, py, s1x, s1y, s2x, s2y, eps)
            if intersect[0][0] != intersect[1][0] or intersect[0][1] != intersect[1][1]:
                i1x = intersect[0, 0]
                i1y = intersect[0, 1]
                u1 = (((i1x - s1x) * (s2x - s1x)) + ((i1y - s1y) * (s2y - s1y))) / segl2

                i2x = intersect[1, 0]
                i2y = intersect[1, 1]
                u2 = (((i2x - s1x) * (s2x - s1x)) + ((i2y - s1y) * (s2y - s1y))) / segl2
                ordered_point = sorted((0, 1, u1, u2))
                lf = ordered_point[1:3]
            else:
                if px == s1x and py==s1y:
                    lf = [0, 0]
                elif px == s2x and py==s2y:
                     lf = [1, 1]
                else:
                    i1x = intersect[0][0]
                    i1y = intersect[0][1]
                    u1 = (((i1x - s1x) * (s2x - s1x)) + ((i1y - s1y) * (s2y - s1y))) / segl2
                    if 0 <= u1 <= 1:
                        lf = [u1, u1]
                    else:
                        lf = [-1, -1]
    return lf


def LF_BF(P, Q, p, q, eps):
    """
    Usage
    -----
    Compute all the free space on the boundary of cells in the diagram for polygonal chains P and Q and the given eps
    LF[(i,j)] is the free space of segment [Pi,Pi+1] from point  Qj
    BF[(i,j)] is the free space of segment [Qj,Qj+1] from point Pj

    Parameters
    ----------
    param P : px2 numpy_array, Trajectory P
    param Q : qx2 numpy_array, Trajectory Q
    param p : float, number of points in Trajectory P
    param q : float, number of points in Trajectory Q
    param eps : float, reachability distance

    Returns
    -------
    LF : dict, free spaces of segments of P from points of Q
    BF : dict, free spaces of segments of Q from points of P
    """
    LF = {}
    for j in range(q):
        for i in range(p - 1):
            LF.update({(i, j): free_line(Q[j], eps, P[i:i + 2])})
    BF = {}
    for j in range(q - 1):
        for i in range(p):
            BF.update({(i, j): free_line(P[i], eps, Q[j:j + 2])})
    return LF, BF


def LR_BR(LF, BF, p, q):
    """
    Usage
    -----
    Compute all the free space,that are reachable from the origin (P[0,0],Q[0,0]) on the boundary of cells
    in the diagram for polygonal chains P and Q and the given free spaces LR and BR

    LR[(i,j)] is the free space, reachable from the origin, of segment [Pi,Pi+1] from point  Qj
    BR[(i,j)] is the free space, reachable from the origin, of segment [Qj,Qj+1] from point Pj

    Parameters
    ----------
    LF : dict, free spaces of segments of P from points of Q
    BF : dict, free spaces of segments of Q from points of P
    param p : float, number of points in Trajectory P
    param q : float, number of points in Trajectory Q

    Returns
    -------
    rep : bool, return true if frechet distance is inf to eps
    LR : dict, is the free space, reachable from the origin, of segments of P from points of Q
    BR : dict, is the free space, reachable from the origin, of segments of Q from points of P
    """
    if not (LF[(0, 0)][0] <= 0 and BF[(0, 0)][0] <= 0 and LF[(p - 2, q - 1)][1] >= 1 and BF[(p - 1, q - 2)][1] >= 1 ):
        rep = False
        BR={}
        LR={}
    else:
        LR = {(0, 0): True}
        BR = {(0, 0): True}
        for i in range(1, p - 1):
            if LF[(i, 0)] != [-1, -1] and LF[(i - 1, 0)] == [0, 1]:
                LR[(i, 0)] = True
            else:
                LR[(i, 0)] = False
        for j in range(1, q - 1):
            if BF[(0, j)] != [-1, -1] and BF[(0, j - 1)] == [0, 1]:
                BR[(0, j)] = True
            else:
                BR[(0, j)] = False
        for i in range(p - 1):
            for j in range(q - 1):
                if LR[(i, j)] or BR[(i, j)]:
                    if LF[(i, j + 1)] != [-1, -1]:
                        LR[(i, j + 1)] = True
                    else:
                        LR[(i, j + 1)] = False
                    if BF[(i + 1, j)] != [-1, -1]:
                        BR[(i + 1, j)] = True
                    else:
                        BR[(i + 1, j)] = False
                else:
                    LR[(i, j + 1)] = False
                    BR[(i + 1, j)] = False
        rep = BR[(p - 2, q - 2)] or LR[(p - 2, q - 2)]
    return rep, LR, BR


def decision_problem(P, Q, p, q, eps):
    """
    Usage
    -----
    Test is the frechet distance between trajectories P and Q are inferior to eps

    Parameters
    ----------
    param P : px2 numpy_array, Trajectory P
    param Q : qx2 numpy_array, Trajectory Q
    param p : float, number of points in Trajectory P
    param q : float, number of points in Trajectory Q
    param eps : float, reachability distance

    Returns
    -------
    rep : bool, return true if frechet distance is inf to eps
    """
    LF, BF = LF_BF(P, Q, p, q, eps)
    rep, _, _ = LR_BR(LF, BF, p, q)
    return rep


def compute_critical_values(P, Q, p, q):
    """
    Usage
    -----
    Compute all the critical values between trajectories P and Q

    Parameters
    ----------
    param P : px2 numpy_array, Trajectory P
    param Q : qx2 numpy_array, Trajectory Q
    param p : float, number of points in Trajectory P
    param q : float, number of points in Trajectory Q

    Returns
    -------
    cc : list, all critical values between trajectories P and Q
    """
    origin = eucl_dist(P[0], Q[0])
    end = eucl_dist(P[-1], Q[-1])
    end_point = max(origin, end)
    cc = set([end_point])
    for i in range(p - 1):
        for j in range(q - 1):
            Lij = point_to_seg(Q[j], P[i], P[i + 1])
            if Lij > end_point:
                cc.add(Lij)
            Bij = point_to_seg(P[i], Q[j], Q[j + 1])
            if Bij > end_point:
                cc.add(Bij)
    return sorted(list(cc))


def frechet(P, Q):
    """
    Usage
    -----
    Compute the frechet distance between trajectories P and Q

    Parameters
    ----------
    param P : px2 numpy_array, Trajectory P
    param Q : qx2 numpy_array, Trajectory Q

    Returns
    -------
    frech : float, the frechet distance between trajectories P and Q
    """
    p = len(P)
    q = len(Q)

    cc = compute_critical_values(P, Q, p, q)
    eps=cc[0]
    while (len(cc) != 1):
        m_i = len(cc) / 2 - 1
        eps = cc[m_i]
        rep = decision_problem(P, Q, p, q, eps)
        if rep:
            cc = cc[:m_i + 1]
        else:
            cc = cc[m_i + 1:]
    frech = eps
    return frech
