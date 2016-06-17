STUFF = "Hi"

from libc.math cimport acos
from libc.math cimport sin
from libc.math cimport fmin
from libc.math cimport fmax
from libc.math cimport sqrt

from basic_euclidean import c_eucl_dist
cimport numpy as np
import numpy as np

cdef float PI = 3.14159265
cdef float HPI = 3.14159265/2

cdef double _ordered_mixed_distance(double six,double siy,double eix,double eiy,double sjx,double sjy,double ejx,
                                    double ejy,double sieix,double sieiy,double sjejx,double sjejy,
                                    double siei_norm_2,double sjej_norm_2):

    cdef double siei_norm,sjej_norm,sisjx,sisjy,siejx,siejy,u1,u2,psx,psy,pex,pey,cos_theta

    siei_norm=sqrt(siei_norm_2)
    sjej_norm=sqrt(sjej_norm_2)
    sisjx=sjx-six
    sisjy=sjy-siy
    siejx=ejx-six
    siejy=ejy-siy


    u1=(sisjx*sieix+sisjy*sieiy)/siei_norm_2
    u2=(siejx*sieix+siejy*sieiy)/siei_norm_2

    psx=six+u1*sieix
    psy=siy+u1*sieiy
    pex=six+u2*sieix
    pey=siy+u2*sieiy

    cos_theta = fmax(-1,fmin(1,(sjejx*sieix+sjejy*sieiy)/(siei_norm*sjej_norm)))
    theta = acos(cos_theta)

    #perpendicular distance
    lpe1=c_eucl_dist(sjx,sjy,psx,psy)
    lpe2=c_eucl_dist(ejx,ejy,pex,pey)
    if lpe1==0 and lpe2==0:
        dped= 0
    else:
        dped = (lpe1*lpe1+lpe2*lpe2)/(lpe1+lpe2)


    #parallel_distance
    lpa1=fmin(c_eucl_dist(six,siy,psx,psy),c_eucl_dist(eix,eiy,psx,psy))
    lpa2=fmin(c_eucl_dist(six,siy,pex,pey),c_eucl_dist(eix,eiy,pex,pey))
    dpad=fmin(lpa1,lpa2)

    #angle_distance
    if 0 <= theta <HPI :
        dad=sjej_norm * sin(theta)
    elif HPI <= theta <= PI:
        dad=sjej_norm
    else:
        raise ValueError("WRONG THETA")
    fdist = (dped+dpad+dad)/3

    return fdist


cdef double _mixed_distance(double six,double siy,double eix,double eiy,double sjx,double sjy,double ejx,double ejy):

    cdef double sieix,sieiy,sjejx,sjejy,siei_norm_2,sjej_norm_2,md

    sieix=eix-six
    sieiy=eiy-siy
    sjejx=ejx-sjx
    sjejy=ejy-sjy


    siei_norm_2=(sieix*sieix)+(sieiy*sieiy)
    sjej_norm_2=(sjejx*sjejx)+(sjejy*sjejy)

    if sjej_norm_2 > siei_norm_2 :
        md=_ordered_mixed_distance(sjx,sjy,ejx,ejy,six,siy,eix,eiy,sjejx,sjejy,sieix,sieiy,sjej_norm_2,siei_norm_2)
    else :
        md=_ordered_mixed_distance(six,siy,eix,eiy,sjx,sjy,ejx,ejy,sieix,sieiy,sjejx,sjejy,siei_norm_2,sjej_norm_2)

    return md

def c_segments_distance(np.ndarray[np.float64_t,ndim=2] traj_0,np.ndarray[np.float64_t,ndim=2] traj_1):

    cdef np.ndarray[np.float64_t,ndim=2] M
    cdef int n0,n1,i,j


    n0=len(traj_0)
    n1=len(traj_1)
    M=np.zeros((n0-1,n1-1))
    for i in range(n0-1):
        for j in range(n1-1):
            M[i,j]=_mixed_distance(traj_0[i,0],traj_0[i,1],traj_0[i+1,0],traj_0[i+1,1],
                                   traj_1[j,0],traj_1[j,1],traj_1[j+1,0],traj_1[j+1,1])
    return M





