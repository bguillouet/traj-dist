STUFF = "Hi"

#cython: boundscheck=False
#cython: wraparound=False

cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free

from libc.math cimport fmin
from basic_euclidean import c_eucl_dist
from cpython cimport bool
from numpy.math cimport INFINITY



cdef double _owd_grid_brut(np.ndarray[long,ndim=2] traj_cell_1, np.ndarray[long,ndim=2] traj_cell_2):

    cdef double D,d
    cdef int n1,n2,p1x,p1y,p2x,p2y,i,j
    cdef np.ndarray[long,ndim=1] p1,p2

    D=0
    n1=len(traj_cell_1)
    n2=len(traj_cell_2)

    for i from 0 <= i < n1:
        p1=traj_cell_1[i]
        p1x=p1[0]
        p1y=p1[1]
        d=INFINITY
        for j from 0 <= j < n2:
            p2=traj_cell_2[j]
            p2x=p2[0]
            p2y=p2[1]
            d=fmin(d,c_eucl_dist(p1x,p1y,p2x,p2y))
        D = D + d
    D = D/n1
    return D

def c_owd_grid_brut(np.ndarray[long,ndim=2] traj_cell_1, np.ndarray[long,ndim=2] traj_cell_2):

    cdef double D,d
    cdef int n1,n2,p1x,p1y,p2x,p2y,i,j
    cdef np.ndarray[long,ndim=1] p1,p2

    D=0
    n1=len(traj_cell_1)
    n2=len(traj_cell_2)

    for i from 0 <= i < n1:
        p1=traj_cell_1[i]
        p1x=p1[0]
        p1y=p1[1]
        d=INFINITY
        for j from 0 <= j < n2:
            p2=traj_cell_2[j]
            p2x=p2[0]
            p2y=p2[1]
            d=fmin(d,c_eucl_dist(p1x,p1y,p2x,p2y))
        D = D + d
    D = D/n1
    return D


cdef double _sowd_grid_brut(np.ndarray[long,ndim=2] traj_cell_1,np.ndarray[long,ndim=2] traj_cell_2):

    cdef double sowd_brut_dist

    sowd_brut_dist = _owd_grid_brut(traj_cell_1,traj_cell_2)+_owd_grid_brut(traj_cell_2,traj_cell_1)
    return sowd_brut_dist/2

def c_sowd_grid_brut(np.ndarray[long,ndim=2] traj_cell_1,np.ndarray[long,ndim=2] traj_cell_2):

    cdef double sowd_brut_dist

    sowd_brut_dist = _owd_grid_brut(traj_cell_1,traj_cell_2)+_owd_grid_brut(traj_cell_2,traj_cell_1)
    return sowd_brut_dist/2



cdef np.ndarray[long,ndim=1] _find_first_min_points( np.ndarray[np.float64_t,ndim=1] pt, int n):

    cdef np.ndarray[np.uint8_t,ndim=1,cast=True] min_points
    cdef np.ndarray[long,ndim=1] min_points_index
    cdef double pti,ptip,ptim
    cdef bool m_p,ciip,cimi

    min_points = np.empty([n],dtype=bool)

    if n == 1:
        min_points[0] = True
    else:
        pti=pt[0]
        ptip=pt[1]
        ciip=  pti < ptip
        min_points[0] = ciip
        for i from 1 <= i < n-1:
            pti = pt[i]
            ptip = pt[i+1]
            ptim = pt[i-1]
            ciip = pti < ptip
            cimi = pti < ptim
            m_p =  (ciip and cimi)
            min_points [i] = m_p
        ptim = pt[n - 2]
        pti = pt[n - 1]
        cimi =  pti < ptim
        min_points[n-1] = cimi

    min_points_index = np.where(np.array(min_points))[0]
    return min_points_index


def c_find_first_min_points( np.ndarray[np.float64_t,ndim=1] pt, int n):

    cdef np.ndarray[np.uint8_t,ndim=1,cast=True] min_points
    cdef np.ndarray[long,ndim=1] min_points_index
    cdef double pti,ptip,ptim
    cdef bool m_p,ciip,cimi

    min_points = np.empty([n],dtype=bool)

    if n == 1:
        min_points[0] = True
    else:
        pti=pt[0]
        ptip=pt[1]
        ciip=  pti < ptip
        min_points[0] = ciip
        for i from 1 <= i < n-1:
            pti = pt[i]
            ptip = pt[i+1]
            ptim = pt[i-1]
            ciip = pti < ptip
            cimi = pti < ptim
            m_p =  (ciip and cimi)
            min_points [i] = m_p
        ptim = pt[n - 2]
        pti = pt[n - 1]
        cimi =  pti < ptim
        min_points[n-1] = cimi

    min_points_index = np.where(np.array(min_points))[0]
    return min_points_index


def c_owd_grid( np.ndarray[long,ndim=2] traj_cell_1,np.ndarray[long,ndim=2] traj_cell_2):

    cdef int n1,n2,px,py,p_precx,p_precy,p2x,p2y,i,j,n_S_old,ig,pgx,pgy,rmin,rmax,pgpx,pgpy,pgpmx,pgpmy,pgppx,pgppy
    cdef double dpp2,D,d,dist,dist_back,dist_forw
    cdef np.ndarray[long,ndim=1] p,p_prec,pg,p2,S_old,pgp,pgpp,pgpm,S_ind
    cdef np.ndarray[np.float64_t,ndim=1] p_t2

    n1 = len(traj_cell_1)
    n2 = len(traj_cell_2)

    p = traj_cell_1[0]
    px = p[0]
    py = p[1]

    p2 = traj_cell_2[0]
    p2x=p2[0]
    p2y=p2[1]

    p_t2 = np.empty([n2],dtype=float)
    dpp2=c_eucl_dist(px,py,p2x,p2y)
    p_t2[0] = dpp2
    for i from 1 <= i < n2:
        p2 = traj_cell_2[i]
        p2x=p2[0]
        p2y=p2[1]
        dpp2_=c_eucl_dist(px,py,p2x,p2y)
        dpp2=fmin(dpp2,dpp2_)
        p_t2[i] = dpp2_

    S_old = _find_first_min_points(p_t2, n2)
    D = dpp2
    for i from 1 <= i < n1:
        p_prec = p
        p_precx=px
        p_precy=py
        p = traj_cell_1[i]
        px=p[0]
        py=p[1]
        S_ind = np.zeros([n2],dtype=int)
        d = INFINITY
        n_S_old=len(S_old)
        for j from 0 <= j < n_S_old:
            ig = S_old[j]
            pg = traj_cell_2[ig]
            pgx=pg[0]
            pgy=pg[1]
            if (p_precy == py and pgx != p_precx) or (p_precx == px and pgy != p_precy):
                S_ind[ig] = 1
                dppg=c_eucl_dist(px,py,pgx,pgy)
                d=fmin(d,dppg)
            else:
                if j == 0:
                    if n_S_old == 1 :
                        rmin = 0
                        rmax = n2
                    else:
                        rmin = 0
                        rmax = S_old[j+1]
                elif j == n_S_old - 1:
                    rmin = S_old[j-1]
                    rmax = n2
                else:
                    rmin=S_old[j-1]+1
                    rmax= S_old[j+1]
                for igp from rmin <= igp < rmax:
                    pgp = traj_cell_2[igp]
                    pgpx = pgp[0]
                    pgpy = pgp[1]
                    if (p_precy == py and pgpx == px ) or (p_precx == px and pgpy == py) or igp==ig  :
                        if igp !=0:
                            pgpm = traj_cell_2[igp-1]
                            pgpmx=pgpm[0]
                            pgpmy=pgpm[1]
                            dist_back=c_eucl_dist(pgpmx,pgpmy,px,py)
                        else:
                            dist_back = INFINITY
                        if igp !=n2-1:
                            pgpp = traj_cell_2[igp+1]
                            pgppx=pgpp[0]
                            pgppy=pgpp[1]
                            dist_forw=c_eucl_dist(pgppx,pgppy,px,py)
                        else:
                            dist_forw = INFINITY
                        dist = c_eucl_dist(pgpx,pgpy,px,py)
                        if dist < dist_back and dist < dist_forw:
                            if not S_ind[igp]:
                                S_ind[igp] = 1
                                d=fmin(d,dist)

        S_old=np.where(S_ind)[0]
        D += d
    return D/n1


cdef double _owd_grid( np.ndarray[long,ndim=2] traj_cell_1,np.ndarray[long,ndim=2] traj_cell_2):

    cdef int n1,n2,px,py,p_precx,p_precy,p2x,p2y,i,j,n_S_old,ig,pgx,pgy,rmin,rmax,pgpx,pgpy,pgpmx,pgpmy,pgppx,pgppy
    cdef double dpp2,D,d,dist,dist_back,dist_forw
    cdef np.ndarray[long,ndim=1] p,p_prec,pg,p2,S_old,pgp,pgpp,pgpm,S_ind
    cdef np.ndarray[np.float64_t,ndim=1] p_t2

    n1 = len(traj_cell_1)
    n2 = len(traj_cell_2)

    p = traj_cell_1[0]
    px = p[0]
    py = p[1]

    p2 = traj_cell_2[0]
    p2x=p2[0]
    p2y=p2[1]

    p_t2 = np.empty([n2],dtype=float)
    dpp2=c_eucl_dist(px,py,p2x,p2y)
    p_t2[0] = dpp2
    for i from 1 <= i < n2:
        p2 = traj_cell_2[i]
        p2x=p2[0]
        p2y=p2[1]
        dpp2_=c_eucl_dist(px,py,p2x,p2y)
        dpp2=fmin(dpp2,dpp2_)
        p_t2[i] = dpp2_

    S_old = _find_first_min_points(p_t2, n2)
    D = dpp2
    for i from 1 <= i < n1:
        p_prec = p
        p_precx=px
        p_precy=py
        p = traj_cell_1[i]
        px=p[0]
        py=p[1]
        S_ind = np.zeros([n2],dtype=int)
        d = INFINITY
        n_S_old=len(S_old)
        for j from 0 <= j < n_S_old:
            ig = S_old[j]
            pg = traj_cell_2[ig]
            pgx=pg[0]
            pgy=pg[1]
            if (p_precy == py and pgx != p_precx) or (p_precx == px and pgy != p_precy):
                S_ind[ig] = 1
                dppg=c_eucl_dist(px,py,pgx,pgy)
                d=fmin(d,dppg)
            else:
                if j == 0:
                    if n_S_old == 1 :
                        rmin = 0
                        rmax = n2
                    else:
                        rmin = 0
                        rmax = S_old[j+1]
                elif j == n_S_old - 1:
                    rmin = S_old[j-1]
                    rmax = n2
                else:
                    rmin=S_old[j-1]+1
                    rmax= S_old[j+1]
                for igp from rmin <= igp < rmax:
                    pgp = traj_cell_2[igp]
                    pgpx = pgp[0]
                    pgpy = pgp[1]
                    if (p_precy == py and pgpx == px ) or (p_precx == px and pgpy == py) or igp==ig  :
                        if igp !=0:
                            pgpm = traj_cell_2[igp-1]
                            pgpmx=pgpm[0]
                            pgpmy=pgpm[1]
                            dist_back=c_eucl_dist(pgpmx,pgpmy,px,py)
                        else:
                            dist_back = INFINITY
                        if igp !=n2-1:
                            pgpp = traj_cell_2[igp+1]
                            pgppx=pgpp[0]
                            pgppy=pgpp[1]
                            dist_forw=c_eucl_dist(pgppx,pgppy,px,py)
                        else:
                            dist_forw = INFINITY
                        dist = c_eucl_dist(pgpx,pgpy,px,py)
                        if dist < dist_back and dist < dist_forw:
                            if not S_ind[igp]:
                                S_ind[igp] = 1
                                d=fmin(d,dist)

        S_old=np.where(S_ind)[0]
        D += d
    return D/n1



cdef double _sowd_grid(np.ndarray[long,ndim=2] traj_cell_1,np.ndarray[long,ndim=2] traj_cell_2):

    cdef double sowd_dist

    sowd_dist = _owd_grid(traj_cell_1,traj_cell_2)+_owd_grid(traj_cell_2,traj_cell_1)
    return sowd_dist/2

def c_sowd_grid(np.ndarray[long,ndim=2] traj_cell_1,np.ndarray[long,ndim=2] traj_cell_2):

    cdef double sowd_dist

    sowd_dist = _owd_grid(traj_cell_1,traj_cell_2)+_owd_grid(traj_cell_2,traj_cell_1)
    return sowd_dist/2