#cython: boundscheck=False
#cython: wraparound=False

STUFF = "Hi"

from libc.math cimport sqrt
from libc.math cimport fmin
cimport numpy as np
import numpy as np
from cpython cimport bool



cdef double _eucl_dist(double x1,double y1,double x2,double y2):
    """
    Usage
    -----
    L2-norm between point (x1,y1) and (x2,y2)

    Parameters
    ----------
    param x1 : float
    param y1 : float
    param x2 : float
    param y2 : float

    Returns
    -------
    dist : float
           L2-norm between (x1,y1) and (x2,y2)
    """
    cdef double d,dx,dy
    dx=(x1-x2)
    dy=(y1-y2)
    dist=sqrt(dx*dx+dy*dy)
    return dist


def c_eucl_dist(double x1,double y1,double x2,double y2):
    """
    Usage
    -----
    L2-norm between point (x1,y1) and (x2,y2)

    Parameters
    ----------
    param x1 : float
    param y1 : float
    param x2 : float
    param y2 : float

    Returns
    -------
    dist : float
           L2-norm between (x1,y1) and (x2,y2)
    """
    cdef double d,dx,dy
    dx=(x1-x2)
    dy=(y1-y2)
    d=sqrt(dx*dx+dy*dy)
    return d



cdef double _point_to_seg(double px, double py, double p1x, double p1y, double p2x, double p2y ):
    """
    Usage
    -----
    Point to segment distance between point (px, py) and segment delimited by (p1x, p1y) and (p2x, p2y)

    Parameters
    ----------
    param px : float, abscissa of the point
    param py : float, ordinate of the point
    param p1x : float, abscissa of the first end point of the segment
    param p1y : float, ordinate of the first end point of the segment
    param p2x : float, abscissa of the second end point of the segment
    param p2y : float, ordinate of the second end point of the segment


    Returns
    -------
    dpl: float
         Point to segment distance between p and s
    """

    cdef double segl,u1,u,ix,iy,dpl
    if p1x==p2x and p1y==p2y:
        dpl=_eucl_dist(px,py,p1x,p1y)
    else:
        segl= _eucl_dist(p1x,p1y,p2x,p2y)
        u1 = (((px - p1x) * (p2x - p1x)) + ((py - p1y) * (p2y - p1y)))
        u = u1 / (segl * segl)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = _eucl_dist(px,py,p1x,p1y)
            iy = _eucl_dist(px,py,p2x,p2y)
            if ix > iy:
                dpl = iy
            else:
                dpl = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = p1x + u * (p2x - p1x)
            iy = p1y + u * (p2y - p1y)
            dpl = _eucl_dist(px,py,ix,iy)

    return dpl

def c_point_to_seg(double px, double py, double p1x, double p1y, double p2x, double p2y ):
    """
    Usage
    -----
    Point to segment distance between point (px, py) and segment delimited by (p1x, p1y) and (p2x, p2y)

    Parameters
    ----------
    param px : float, abscissa of the point
    param py : float, ordinate of the point
    param p1x : float, abscissa of the first end point of the segment
    param p1y : float, ordinate of the first end point of the segment
    param p2x : float, abscissa of the second end point of the segment
    param p2y : float, ordinate of the second end point of the segment


    Returns
    -------
    dpl: float
         Point to segment distance between p and s
    """
    cdef double segl,u1,u,ix,iy,dpl
    if p1x==p2x and p1y==p2y:
        dpl=_eucl_dist(px,py,p1x,p1y)
    else:
        segl= _eucl_dist(p1x,p1y,p2x,p2y)
        u1 = (((px - p1x) * (p2x - p1x)) + ((py - p1y) * (p2y - p1y)))
        u = u1 / (segl * segl)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = _eucl_dist(px,py,p1x,p1y)
            iy = _eucl_dist(px,py,p2x,p2y)
            if ix > iy:
                dpl = iy
            else:
                dpl = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = p1x + u * (p2x - p1x)
            iy = p1y + u * (p2y - p1y)
            dpl = _eucl_dist(px,py,ix,iy)

    return dpl

cdef double _point_to_trajectory (double px,double py, np.ndarray[np.float64_t,ndim=2] t):
    """
    Usage
    -----
    Point-to-trajectory distance between point (px, py) and the trajectory t.
    The Point to trajectory distance is the minimum of point-to-segment distance between p and all segment s of t

    Parameters
    ----------
    param px : float, abscissa of the point
    param py : float, ordinate of the point
    param t : len(t)x2 numpy_array

    Returns
    -------
    dpt : float,
          Point-to-trajectory distance between p and trajectory t
    """
    cdef double dpt,dist_j0
    cdef int nt,i
    cdef np.ndarray[np.float64_t,ndim=1] xt,yt

    nt=len(t)
    xt=t[:,0]
    yt=t[:,1]
    dist_j0=9e100
    for i from 0 <= i < (nt-1):
        dist_j0=fmin(dist_j0,_point_to_seg(px,py,xt[i],yt[i],xt[i+1],yt[i+1]))

    return dist_j0

def c_point_to_trajectory (double px,double py, np.ndarray[np.float64_t,ndim=2] t):
    """
    Usage
    -----
    Point-to-trajectory distance between point (px, py) and the trajectory t.
    The Point to trajectory distance is the minimum of point-to-segment distance between p and all segment s of t

    Parameters
    ----------
    param px : float, abscissa of the point
    param py : float, ordinate of the point
    param t : len(t)x2 numpy_array

    Returns
    -------
    dpt : float,
          Point-to-trajectory distance between p and trajectory t
    """
    cdef double dpt,dist_j0
    cdef int nt,i
    cdef np.ndarray[np.float64_t,ndim=1] xt,yt

    nt=len(t)
    xt=t[:,0]
    yt=t[:,1]
    dist_j0=9e100
    for i from 0 <= i < (nt-1):
        dist_j0=fmin(dist_j0,_point_to_seg(px,py,xt[i],yt[i],xt[i+1],yt[i+1]))

    return dist_j0

cdef np.ndarray[np.float64_t,ndim=2] _circle_line_intersection(double px, double py, double s1x, double s1y,
double
s2x, double s2y, double eps):
    """
    Usage
    -----
    Find the intersections between the circle of radius eps and center (px, py) and the line delimited by points
    (s1x, s1y) and (s2x, s2y).
    It is supposed here that the intersection between them exists. If no, raise error

    Parameters
    ----------
    param px : float, centre's abscissa of the circle
    param py : float, centre's ordinate of the circle
    eps : float, radius of the circle
    s1x : abscissa of the first point of the line
    s1y : ordinate of the first point of the line
    s2x : abscissa of the second point of the line
    s2y : ordinate of the second point of the line

    Returns
    -------
    intersect : 2x2 numpy_array
                Coordinate of the two intersections.
                If the two intersections are the same, that's means that the line is a tangent of the circle.
    """
    cdef double rac,y1,y2,m,c,A,B,C,delta,x,y,sdelta,x1,x2
    cdef np.ndarray[np.float64_t,ndim=2] intersect
    cdef bool delta_strict_inf_0,delta_inf_0,delta_strict_sup_0

    if s2x==s1x :
        rac=sqrt((eps*eps) - ((s1x-px)*(s1x-px)))
        y1 = py+rac
        y2 = py-rac
        intersect = np.array([[s1x,y1],[s1x,y2]])
    else:
        m= (s2y-s1y)/(s2x-s1x)
        c= s2y-m*s2x
        A=m*m+1
        B=2*(m*c-m*py-px)
        C=py*py-eps*eps+px*px-2*c*py+c*c
        delta=B*B-4*A*C
        delta_inf_0 = delta <= 0
        delta_strict_sup_0 = delta > 0
        if delta_inf_0 :
            delta_strict_inf_0 = delta <0
            x = -B/(2*A)
            y = m*x+c
            intersect = np.array([[x,y],[x,y]])
        elif delta_strict_sup_0 > 0 :
            sdelta = sqrt(delta)
            x1= (-B+sdelta)/(2*A)
            y1=m*x1+c
            x2= (-B-sdelta)/(2*A)
            y2=m*x2+c
            intersect = np.array([[x1,y1],[x2,y2]])
        else :
            raise ValueError("The intersection between circle and line is supposed to exist")
    return intersect

def c_circle_line_intersection(double px, double py, double s1x, double s1y, double
s2x, double s2y, double eps):
    """
    Usage
    -----
    Find the intersections between the circle of radius eps and center (px, py) and the line delimited by points
    (s1x, s1y) and (s2x, s2y).
    It is supposed here that the intersection between them exists. If no, raise error

    Parameters
    ----------
    param px : float, centre's abscissa of the circle
    param py : float, centre's ordinate of the circle
    param eps : float, radius of the circle
    param s1x : abscissa of the first point of the line
    param s1y : ordinate of the first point of the line
    param s2x : abscissa of the second point of the line
    param s2y : ordinate of the second point of the line

    Returns
    -------
    intersect : 2x2 numpy_array
                Coordinate of the two intersections.
                If the two intersections are the same, that's means that the line is a tangent of the circle.
    """
    cdef double rac,y1,y2,m,c,A,B,C,delta,x,y,sdelta,x1,x2
    cdef np.ndarray[np.float64_t,ndim=2] intersect
    cdef bool delta_strict_inf_0,delta_inf_0,delta_strict_sup_0

    if s2x==s1x :
        rac=sqrt((eps*eps) - ((s1x-px)*(s1x-px)))
        y1 = py+rac
        y2 = py-rac
        intersect = np.array([[s1x,y1],[s1x,y2]])
    else:
        m= (s2y-s1y)/(s2x-s1x)
        c= s2y-m*s2x
        A=m*m+1
        B=2*(m*c-m*py-px)
        C=py*py-eps*eps+px*px-2*c*py+c*c
        delta=B*B-4*A*C
        delta_inf_0 = delta <= 0
        delta_strict_sup_0 = delta > 0
        if delta_inf_0 :
            delta_strict_inf_0 = delta <0
            x = -B/(2*A)
            y = m*x+c
            intersect = np.array([[x,y],[x,y]])
        elif delta_strict_sup_0 > 0 :
            sdelta = sqrt(delta)
            x1= (-B+sdelta)/(2*A)
            y1=m*x1+c
            x2= (-B-sdelta)/(2*A)
            y2=m*x2+c
            intersect = np.array([[x1,y1],[x2,y2]])
        else :
            raise ValueError("The intersection between circle and line is supposed to exist")
    return intersect