import math

import numpy as np
from sklearn.metrics.pairwise import haversine_distances

from .basic_spherical_vectorized import points_to_trajectory_vectorized, closest_point_to_trajectory_vectorized, \
    distances_of_points_to_trajectory_vectorized
from ..constants import earth_radius
from ..trajectory_utils import increasing

R = earth_radius()

######################
# Spherical Geometry #
######################


def calculate_distance(lon_values: np.array, lat_values: np.array) -> np.array:
    """distance between points in meters - haversine function """
    long1 = np.radians(lon_values[:-1])
    long2 = np.radians(lon_values[1:])
    lat1 = np.radians(lat_values[:-1])
    lat2 = np.radians(lat_values[1:])

    distance = ((np.sin(0.5*(lat1 - lat2)) ** 2)
                + (np.cos(lat1)
                   * np.cos(lat2)
                   * np.sin((long1 - long2) * 0.5) ** 2))
    distance = np.arcsin(np.sqrt(distance)) * 2 * R
    distance = np.insert(distance, 0, np.nan)
    return distance


def s_spd_vectorized(t1, t2, t2_dist, dist_t1_t2):
    """ spd distance of t1 from t2

    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array - order [lon, lat]
    param t2 :  len(t2)x2 numpy_array - order [lon, lat]
    """
    distances, distance_indices = points_to_trajectory_vectorized(t1, t2, dist_t1_t2, t2_dist)
    spd = np.mean(distances)
    return spd


def s_spd_vectorized_distances(t1, t2, t2_dist=None):
    """ shortest distances of each pt in t1 from t2

    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array - order [lon, lat]
    param t2 :  len(t2)x2 numpy_array - order [lon, lat]
    param t2_dist: len(t2)-1x1 float ndarray - consecutive distances among t2 points

    Returns
    ----------
    param distances: list of the shortest distances of each pt in t1 to t2
    param same_direction: bool True iff t1 and t2 are in the same direction

    """
    if t2_dist is None:
        t2_dist = calculate_distance(t2[:, 0], t2[:, 1])[1:]
    t1_lat_lon = t1[:, ::-1]  # (lat, lon) order
    t2_lat_lon = t2[:, ::-1]  # (lat, lon) order
    dist_t1_t2 = haversine_distances(np.radians(t1_lat_lon), np.radians(t2_lat_lon)) * R
    distances, distance_indices = points_to_trajectory_vectorized(t1, t2, dist_t1_t2, t2_dist)
    same_direction = increasing(distance_indices)
    spd = np.mean(distances)
    return spd, distances, same_direction


def s_sspd_vectorized(t1, t2):
    """
    Usage
    -----
    The sspd-distance between trajectories t1 and t2.
    The sspd-distance is the mean of the spd-distance between of t1 from t2 and the spd-distance of t2 from t1.

    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array - order [lon, lat]
    param t2 :  len(t2)x2 numpy_array - order [lon, lat]

    Returns
    -------
    sspd : float
            sspd-distance of trajectory t2 from trajectory t1
    """
    lats1 = t1[:, 1]
    lons1 = t1[:, 0]
    lats2 = t2[:, 1]
    lons2 = t2[:, 0]

    t1_dist = calculate_distance(lons1, lats1)[1:]
    t2_dist = calculate_distance(lons2, lats2)[1:]

    t1_lat_lon = t1[:, ::-1]  # (lat, lon) order
    t2_lat_lon = t2[:, ::-1]  # (lat, lon) order
    dist_t1_t2 = haversine_distances(np.radians(t1_lat_lon), np.radians(t2_lat_lon)) * R
    spd1 = s_spd_vectorized(t1, t2, t2_dist, dist_t1_t2)
    spd2 = s_spd_vectorized(t2, t1, t1_dist, dist_t1_t2.T)
    dist = spd1 + spd2  # TODO: / 2
    return dist


def s_pt_to_traj_dist_vectorized(p1, t2, t2_dist=None):
    """
    Usage
    -----
    The shortest distance of point p1 from trajectory t2

    Parameters
    ----------
    param p1: [2x1] float ndarray [lon, lat]
    param t2: len(t1)x2 float ndarray - order [lon, lat]

    Returns
    -------
    float
        shortest distance of point p0 from trajectory t1

    """
    if t2_dist is None:
        t2_dist = calculate_distance(t2[:, 0], t2[:, 1])[1:]
    t2_lat_lon = t2[:, ::-1]  # (lat, lon) order
    pt_array = np.array([[p1[1], p1[0]]])
    dist_p1_t2 = haversine_distances(np.radians(pt_array), np.radians(t2_lat_lon)) * R
    distances, _ = points_to_trajectory_vectorized(np.array([p1]), t2, dist_p1_t2, t2_dist)
    return distances[0]


def s_pt_to_traj_dist_threshold_vectorized(dist_threshold, p1, t2, t2_dist=None) -> (float, int, int):
    """
    Usage
    -----
    The shortest distance of point p1 from trajectory t2.
    If it is shorter than *dist_threshold*, return its first occurrence along the path,
    i.e. where point p1 is at least as close as dist_threshold to t2.

    Parameters
    ----------
    param dist_threshold: float
    param p1: [2x1] float ndarray [lon, lat]
    param t2: len(t2)x2 float ndarray - order [lon, lat]

    Returns
    -------
        float
            if such exists: the first distance that is smaller than the threshold
            or: the shortest distance of point p1 from the t2
        int
            the index of the first *segment* of t2 which is closer than dist_threshold to p1, None if not exists
        int
            if such exists, the index of the first point on t2 which is closer than dist_threshold to p1. Otherwise None

    """
    if t2_dist is None:
        t2_dist = calculate_distance(t2[:, 0], t2[:, 1])[1:]
    t2_lat_lon = t2[:, ::-1]  # (lat, lon) order
    pt_array = np.array([[p1[1], p1[0]]])
    dist_p1_t2 = haversine_distances(np.radians(pt_array), np.radians(t2_lat_lon)) * R
    distances = distances_of_points_to_trajectory_vectorized(np.array([p1]), t2, dist_p1_t2, t2_dist)[0]
    distances_p1_t2 = dist_p1_t2[0]
    indices_passed_threshold = np.flatnonzero(distances <= dist_threshold)
    if len(indices_passed_threshold) > 0:
        idx = indices_passed_threshold[0]
        pt_idx = None
        if math.isclose(distances[idx], distances_p1_t2[idx], abs_tol=1e-1, rel_tol=1e-3):
            pt_idx = idx
        elif math.isclose(distances[idx], distances_p1_t2[idx+1], abs_tol=1e-1, rel_tol=1e-3):
            pt_idx = idx + 1
        return distances[idx], idx, pt_idx
    return distances.min(), None, None


def s_closest_pt_vectorized(p1, t2, t2_dist=None):
    """
    Usage
    -----
    Finds a point on trajectory t2 that is closest to p1
    The closest point is either one of t2 trajectory points or the (shortest) projection of p1 on t2

    Parameters
    ----------
    param p1: [2x1] float ndarray [lon, lat]
    param t2: len(t1)x2 float ndarray - order [lon, lat]
    param t2_dist: len(t2)-1x1 float ndarray - consecutive distances among t2 points

    Returns
    -------
    lon: float
    lat: float
    idx: float the point index of t2 which is closest to p1

    """
    if t2_dist is None:
        t2_dist = calculate_distance(t2[:, 0], t2[:, 1])[1:]
    t2_lat_lon = t2[:, ::-1]
    pt_array = np.array([[p1[1], p1[0]]])
    dist_p1_t2 = haversine_distances(np.radians(pt_array), np.radians(t2_lat_lon)) * R
    lon, lat, idx = closest_point_to_trajectory_vectorized(p1, t2, dist_p1_t2, t2_dist)
    return lon, lat, idx
