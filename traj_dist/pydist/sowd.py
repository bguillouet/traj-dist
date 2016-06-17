import numpy as np
import linecell as linec


def owd_grid_brut(traj_cell_1,traj_cell_2):
    """
    Usage
    -----
    The owd-distance of trajectory t2 from trajectory t1

    Parameters
    ----------
    param traj_cell_1 :  len(t1)x2 numpy_array
    param traj_cell_2 :  len(t2)x2 numpy_array

    Returns
    -------
    owd : float
           owd-distance of trajectory t2 from trajectory t1
    """
    D=0
    n=len(traj_cell_1)
    for p1 in traj_cell_1:
        d=map(lambda x : np.linalg.norm(p1-x),traj_cell_2)
        D+=min(d)
    owd = D/n
    return D/n


def find_first_min_points(pt, n):
    """
    Usage
    -----
    Return the index of the min-point in the vector pt of size n.

    Parameters
    ----------
    param pt :  len(t1)x1 numpy_array
    param n :  int

    Returns
    -------
    min_point_index  : nbumber of min points x1 numpy_array

    """
    if n == 1:
        min_points = [True]
    else:
        min_points = [pt[0] < pt[1]]
        for i in range(1, n - 1):
            m_p = pt[i] < pt[i + 1] and pt[i] < pt[i - 1]
            min_points.append(m_p)
        min_points.append(pt[n - 1] < pt[n - 2])
    min_point_index = np.where(np.array(min_points))[0]
    return min_point_index

def owd_grid(traj_cell_1,traj_cell_2):
    """
    Usage
    -----
    The owd-distance of trajectory t2 from trajectory t1

    Parameters
    ----------
    param traj_cell_1 :  len(t1)x2 numpy_array
    param traj_cell_2 :  len(t2)x2 numpy_array

    Returns
    -------
    owd : float
           owd-distance of trajectory t2 from trajectory t1
    """
    n1 = len(traj_cell_1)
    n2 = len(traj_cell_2)

    p = traj_cell_1[0]
    p_t2 = map(lambda x: np.linalg.norm(p - x), traj_cell_2)
    S_old = find_first_min_points(p_t2, n2)
    D = min(p_t2)
    for i in range(1, n1):
        p_prec = p
        p = traj_cell_1[i]
        S = []
        d = []
        n_S_old=len(S_old)
        for j in range(n_S_old):
            ig = S_old[j]
            pg = traj_cell_2[ig]
            if (p_prec[1] == p[1]) and (pg[0] != p_prec[0]) or (p_prec[0] == p[0]) and (pg[1] != p_prec[1]):
                S.append(ig)
                d.append(np.linalg.norm(p - pg))
            else:
                if j == 0:
                    if n_S_old == 1 :
                        ranges = range(0,n2)
                    else:
                        ranges = range(0, S_old[j + 1])
                elif j == n_S_old - 1:
                    ranges = range(S_old[j - 1], n2)
                else:
                    ranges = range(S_old[j - 1] + 1, S_old[j + 1])
                for igp in ranges:
                    pgp = traj_cell_2[igp]
                    if (p_prec[1] == p[1] and pgp[0] == p[0]) or (p_prec[0] == p[0] and pgp[1] == p[1]) or igp==ig :
                        dist_back = np.linalg.norm(traj_cell_2[igp - 1]-p) if igp!=0 else np.inf
                        dist_forw = np.linalg.norm(traj_cell_2[igp + 1]-p) if igp!= n2-1 else np.inf
                        dist = np.linalg.norm(pgp - p)
                        if dist < dist_back and dist < dist_forw:
                            if not (igp in S):
                                S.append(igp)
                                d.append(dist)
        S_old=S
        #p_t2 = map(lambda x: np.linalg.norm(traj_cell_1[i] - x), traj_cell_2)
        #print(np.all(map(lambda x,y:x==y,S_old,find_first_min_points(p_t2, n2))))
        D += min(d)
    return D/n1


def sowd_grid(traj_cell_1,traj_cell_2):
    sowd_dist=owd_grid(traj_cell_1,traj_cell_2)+owd_grid(traj_cell_2,traj_cell_1)
    return sowd_dist/2

def sowd_grid_brut(traj_cell_1,traj_cell_2):
    sowd_brut_dist = owd_grid_brut(traj_cell_1,traj_cell_2)+owd_grid_brut(traj_cell_2,traj_cell_1)
    return sowd_brut_dist/2

def sowd(traj_1,traj_2,precision=7,converted=False):
    if converted:
        d = sowd_grid(traj_1,traj_2)
    else:
        cells_list, _, _ = linec.trajectory_set_grid([traj_1,traj_2], precision)
        d = sowd_grid(cells_list[0],cells_list[1])
    return d

def sowd_brut(traj_1,traj_2,precision=7,converted=False):
    if converted:
        d = sowd_grid_brut(traj_1,traj_2)
    else:
        cells_list, _, _ = linec.trajectory_set_grid([traj_1,traj_2], precision)
        d = sowd_grid_brut(cells_list[0],cells_list[1])
    return d
