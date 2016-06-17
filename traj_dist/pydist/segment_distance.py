from basic_euclidean import eucl_dist
import math
import numpy as np
PI=math.pi
HPI=math.pi/2


def ordered_mixed_distance(si,ei,sj,ej,siei,sjej,siei_norm_2,sjej_norm_2):

    siei_norm=math.sqrt(siei_norm_2)
    sjej_norm=math.sqrt(sjej_norm_2)
    sisj=sj-si
    siej=ej-si

    u1=(sisj[0]*siei[0]+sisj[1]*siei[1])/siei_norm_2
    u2=(siej[0]*siei[0]+siej[1]*siei[1])/siei_norm_2

    ps=si+u1*siei
    pe=si+u2*siei

    cos_theta = max(-1,min(1,(sjej[0]*siei[0]+sjej[1]*siei[1])/(siei_norm*sjej_norm)))
    theta = math.acos(cos_theta)

    #perpendicular distance
    lpe1=eucl_dist(sj,ps)
    lpe2=eucl_dist(ej,pe)
    if lpe1==0 and lpe2==0:
        dped= 0
    else:
        dped = (lpe1*lpe1+lpe2*lpe2)/(lpe1+lpe2)

    #parallel_distance
    lpa1=min(eucl_dist(si,ps),eucl_dist(ei,ps))
    lpa2=min(eucl_dist(si,pe),eucl_dist(ei,pe))
    dpad=min(lpa1,lpa2)

    #angle_distance
    if 0 <= theta <HPI :
        dad=sjej_norm * math.sin(theta)
    elif HPI <= theta <= PI:
        dad=sjej_norm
    else:
        raise ValueError("WRONG THETA")

    fdist = (dped+dpad+dad)/3

    return fdist

def mixed_distance(si,ei,sj,ej):
    siei=ei-si
    sjej=ej-sj

    siei_norm_2=(siei[0]*siei[0])+(siei[1]*siei[1])
    sjej_norm_2=(sjej[0]*sjej[0])+(sjej[1]*sjej[1])

    if sjej_norm_2 > siei_norm_2 :
        md=ordered_mixed_distance(sj,ej,si,ei,sjej,siei,sjej_norm_2,siei_norm_2)
    else :
        md=ordered_mixed_distance(si,ei,sj,ej,siei,sjej,siei_norm_2,sjej_norm_2)

    return md

def segments_distance(traj_0,traj_1):

    n0=len(traj_0)
    n1=len(traj_1)
    M=np.zeros((n0-1,n1-1))
    for i in range(n0-1):
        for j in range(n1-1):
            M[i,j]=mixed_distance(traj_0[i],traj_0[i+1],traj_1[j],traj_1[j+1])
    return M



