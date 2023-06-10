import numpy as np
import math
from scipy.spatial.distance import cdist


def eucl_dist(x, y):
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
    dist = np.linalg.norm(x - y)
    return dist


def eucl_dist_traj(t1, t2):
    """
    Usage
    -----
    High-dimensional European distance between point of trajectories t1 and t2

    Parameters
    ----------
    param t1 : len(t1)xh numpy_array
    param t2 : len(t1)xh numpy_array

    Returns
    -------
    dist : float
           High-dimensional European distance between point of trajectories t1 and t2
    """
    dist = 0
    try:
        for i in range(len(t1)):
            dist += (t1[i] - t2[i]) ** 2
        dist = math.sqrt(dist)
    except:
        ValueError("Trajectories must have the same dimension")
    return dist


def point_to_seg(p, s1, s2):
    """
    Usage
    -----
    Point to segment distance between point p and segment delimited by s1 and s2

    Parameters
    ----------
    param p : 1xh numpy_array
    param s1 : 1xh numpy_array
    param s2 : 1xh numpy_array

    Returns
    -------
    dpl: float
         Point to segment distance between p and s
    """
    line_vector = s2 - s1
    point_vector = p - s1
    line_length = np.linalg.norm(line_vector)
    line_unit_vector = line_vector / line_length
    projection_length = np.dot(point_vector, line_unit_vector)
    projection_vector = projection_length * line_unit_vector
    distance_vector = point_vector - projection_vector
    distance = np.linalg.norm(distance_vector)
    return distance


def point_to_trajectory(p, t):
    """
    Usage
    -----
    Point-to-trajectory distance between point p and the trajectory t.
    The Point to trajectory distance is the minimum of point-to-segment distance between p and all segment s of t

    Parameters
    ----------
    param p: 1xh numpy_array
    param t : l_txh numpy_array

    Returns
    -------
    dpt : float,
          Point-to-trajectory distance between p and trajectory t
    """
    min_distance = float('inf')
    for i in range(len(t) - 1):
        segment_start = t[i]
        segment_end = t[i + 1]
        segment_vector = segment_end - segment_start
        point_vector = p - segment_start
        projection_length = np.dot(point_vector, segment_vector) / np.linalg.norm(segment_vector)
        if 0 <= projection_length <= np.linalg.norm(segment_vector):
            projected_point = segment_start + (projection_length * segment_vector / np.linalg.norm(segment_vector))
            distance = eucl_dist(p, projected_point)
        else:
            distance = min(eucl_dist(p, segment_start), eucl_dist(p, segment_end))
        min_distance = min(min_distance, distance)
    return min_distance


def circle_line_intersection(px, py, s1x, s1y, s2x, s2y, eps):
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
    if s2x == s1x:
        rac = math.sqrt((eps * eps) - ((s1x - px) * (s1x - px)))
        y1 = py + rac
        y2 = py - rac
        intersect = np.array([[s1x, y1], [s1x, y2]])
    else:
        m = (s2y - s1y) / (s2x - s1x)
        c = s2y - m * s2x
        A = m * m + 1
        B = 2 * (m * c - m * py - px)
        C = py * py - eps * eps + px * px - 2 * c * py + c * c
        delta = B * B - 4 * A * C
        if delta <= 0:
            x = -B / (2 * A)
            y = m * x + c
            intersect = np.array([[x, y], [x, y]])
        elif delta > 0:
            sdelta = math.sqrt(delta)
            x1 = (-B + sdelta) / (2 * A)
            y1 = m * x1 + c
            x2 = (-B - sdelta) / (2 * A)
            y2 = m * x2 + c
            intersect = np.array([[x1, y1], [x2, y2]])
        else:
            raise ValueError("The intersection between circle and line is supposed to exist")
    return intersect
