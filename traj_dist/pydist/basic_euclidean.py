import numpy as np
import math

def eucl_dist(x,y):
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
    dist = np.linalg.norm(x-y)
    return dist

def point_to_seg(p,s1,s2):
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
    if p1x==p2x and p1y==p2y:
        dpl=eucl_dist(p,s1)
    else:
        segl= eucl_dist(s1,s2)
        u1 = (((px - p1x) * (p2x - p1x)) + ((py - p1y) * (p2y - p1y)))
        u = u1 / (segl * segl)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = eucl_dist(p,s1)
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

def point_to_trajectory (p, t):
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
    dpt=min(map(lambda s1,s2 : point_to_seg(p,s1,s2), t[:-1],t[1:] ))
    return dpt


def circle_line_intersection(px,py,s1x,s1y,s2x,s2y,eps):
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
    if s2x==s1x :
        rac=math.sqrt((eps*eps) - ((s1x-px)*(s1x-px)))
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
        if delta <= 0 :
            x = -B/(2*A)
            y = m*x+c
            intersect = np.array([[x,y],[x,y]])
        elif delta > 0 :
            sdelta = math.sqrt(delta)
            x1= (-B+sdelta)/(2*A)
            y1=m*x1+c
            x2= (-B-sdelta)/(2*A)
            y2=m*x2+c
            intersect = np.array([[x1,y1],[x2,y2]])
        else :
            raise ValueError("The intersection between circle and line is supposed to exist")
    return intersect
