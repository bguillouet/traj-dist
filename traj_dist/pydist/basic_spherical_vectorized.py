import math
import numpy as np

from traj_dist.constants import earth_radius
from traj_dist.pydist.basic_spherical import cross_track_point

rad = math.pi / 180.0
R = earth_radius()


def initial_bearing_vectorized(lons1, lats1, lons2, lats2):
    """
    Usage
    -----
    Bearing between (lon1,lat1) and (lon2,lat2), in degree.

    Parameters
    ----------
    param lats1: float ndarray, latitude of the first point
    param lons1: float ndarray, longitude of the first point
    param lats2: float ndarray, latitude of the second point
    param lons2: float ndarray, longitude of the second point

    Returns
    -------
    brng: float
           Bearing between (lons1,lats1) and (lons2,lats2), in degree.

    """
    lons2 = lons2.reshape(len(lons2), 1)
    lats2 = lats2.reshape(len(lats2), 1)

    dlon = rad * (lons2 - lons1)
    y = np.sin(dlon) * np.cos(rad * lats2)
    x = np.cos(rad * lats1) * np.sin(rad * lats2) - np.sin(rad * lats1) * np.cos(rad * lats2) * np.cos(dlon)
    ibrng = np.arctan2(y, x)
    return ibrng


def initial_bearing(lons1, lats1, lons2, lats2):
    """
    Usage
    -----
    Bearing between (lon1,lat1) and (lon2,lat2), in degree.

    Parameters
    ----------
    param lats1: float ndarray, latitude of the first point
    param lons1: float ndarray, longitude of the first point
    param lats2: float ndarray, latitude of the second point
    param lons2: float ndarray, longitude of the second point

    Returns
    -------
    brng: float
           Bearing between (lon1,lat1) and (lon2,lat2), in degree.

    """
    dlon = rad * (lons2 - lons1)
    y = np.sin(dlon) * np.cos(rad * lats2)
    x = np.cos(rad * lats1) * np.sin(rad * lats2) - np.sin(rad * lats1) * np.cos(rad * lats2) * np.cos(dlon)
    ibrng = np.arctan2(y, x)

    return ibrng


def cross_track_distance_vectorized(lons1, lats1, lons2, lats2, lons3, lats3, d13):
    """
    Usage
    -----
    Angular cross-track-distance of a point (lon3, lat3) from a great-circle path between (lon1, lat1) and (lon2, lat2)
    The sign of this distance tells which side of the path the third point is on.

    Parameters :
    ----------
    param lats1: float ndarray, latitudes of the first point
    param lons1: float ndarray, longitudes of the first point
    param lats2: float ndarray, latitudes of the second point
    param lons2: float ndarray, longitudes of the second point
    param lats3: float ndarray, latitudes of the third point
    param lons3: float ndarray, longitudes of the third point

    Usage
    -----
    crt: float
         the (angular) cross_track_distance

    """

    theta13 = initial_bearing_vectorized(lons1, lats1, lons3, lats3)  # bearing from start point to third point
    theta12 = initial_bearing(lons1, lats1, lons2, lats2)  # bearing from start point to end point

    crt = np.arcsin(np.sin(d13 / R) * np.sin(theta13 - theta12)) * R

    return crt


def along_track_distance_vectorized(crt, d13):
    """
    Usage
    -----
    The along-track distance from the start point (lon1, lat1) to the closest point on the path
    to the third point (lon3, lat3).

    Parameters
    ----------
    param crt : [n x m] float numpy-array, cross_track_distance
    param d13 : [n x m] float numpy-array, along_track_distance

    Returns
    -------
    alt: [n x m] float numpy-array
         The along-track distance
    """
    res = np.cos(d13 / R) / np.cos(crt / R)
    res = np.minimum(1., res)
    res = np.maximum(-1., res)
    alt = np.arccos(res) * R
    return alt


def points_to_path_vectorized(lons1, lats1, lons2, lats2, lons3, lats3, d13, d23, d12):
    """
    Usage
    -----
    The point-to-path distance between points (lons3, lats3) and paths delimited by (lons1, lats1) and (lons2, lats2).
    The point-to-path distance is the cross_track distance between the great circle path if the projection of
    the third point(s) lies on the path. If it is not on the path, return the minimum of the
    great_circle_distance(s) between the first and the third or the second and the third point(s).

    Parameters
    ----------
    param lons1: float ndarray, longitudes of the first point
    param lats1: float ndarray, latitudes of the first point
    param lons2: float ndarray, longitudes of the second point
    param lats2: float ndarray, latitudes of the second point
    param lons3: float ndarray, longitudes of the third point
    param lats3: float ndarray, latitudes of the third point
    param d13: float [len(lons3) x len(lons1)] ndarray
    param d23: float [len(lons3) x len(lons1)] ndarray
    param d12: float ndarray,

    Returns
    -------
    ptp: float [len(lons3) x len(lons1)] ndarray
        The point-to-path distance between points (lons3, lats3)
        and path delimited by (lons1, lats1) and (lons2, lats2)
    """
    crt = cross_track_distance_vectorized(lons1, lats1, lons2, lats2, lons3, lats3, d13)
    d1p = along_track_distance_vectorized(crt, d13)
    d2p = along_track_distance_vectorized(crt, d23)

    ptp = np.abs(crt)
    tmp = np.tile(d12, (len(d1p), 1))
    mask = (d1p > tmp)
    ptp[mask] = np.minimum(d13, d23)[mask]
    mask = (d2p > tmp)
    ptp[mask] = np.minimum(d13, d23)[mask]
    return ptp


def points_to_trajectory_vectorized(pts, t, dist_pts_t, t_dist):
    """
    Usage
    -----
    The points-to-trajectory distance between points (pts) and trajectory t
    for each point in pts, the shortest distance to trajectory t is calculated

    Parameters
    ----------
    pts - len(t)x2 float ndarray - order [lon, lat]
    t - len(t)x2 float ndarray - order [lon, lat]
    dist_pts_t - [len(t) x len(t)] float ndarray - order [lon, lat]
    t_dist - len(t)-1 float ndarray

    Returns
    -------
    1. float [len(pts)] float ndarray - array of minimum distances of pts to t
    2. [len(pts)] int ndarray - array of the minimum distance indices
    """
    distances = distances_of_points_to_trajectory_vectorized(pts, t, dist_pts_t, t_dist)
    return distances.min(axis=1), distances.argmin(axis=1)


def distances_of_points_to_trajectory_vectorized(pts, t, dist_pts_t, t_dist):
    """
    Usage
    -----
    The points-to-trajectory distance between points (pts) and trajectory t
    for each point in pts, the list of distances to trajectory t is calculated

    Parameters
    ----------
    pts - len(t)x2 float ndarray - order [lon, lat]
    t - len(t)x2 float ndarray - order [lon, lat]
    dist_pts_t - [len(t) x len(t)] float ndarray - order [lon, lat]
    t_dist - len(t)-1 float ndarray

    Returns
    -------
    distances: float [len(pts) x len(t_dist)] ndarray
    """
    lon3, lat3 = pts[:, 0], pts[:, 1]
    d13 = dist_pts_t[:, :-1]
    d23 = dist_pts_t[:, 1:]
    d12 = t_dist
    distances = points_to_path_vectorized(t[:-1, 0], t[:-1, 1], t[1:, 0], t[1:, 1], lon3, lat3, d13, d23, d12)
    return distances


def closest_point_to_trajectory_vectorized(pt, t, dist_pt_t, t_dist):
    distances, distance_indices = points_to_trajectory_vectorized(np.array([pt]), t, dist_pt_t, t_dist)
    min_distance = distances[0]
    min_index = distance_indices[0]
    # allow 0.1 meter absolute difference or 1e-3 relative difference (the min of the 2)
    # abs_tol is defined for numbers close to 0
    if math.isclose(dist_pt_t[0, min_index], min_distance, abs_tol=1e-1, rel_tol=1e-3):
        lon, lat = t[min_index, 0], t[min_index, 1]
    else:
        # project pt on t_closest_segment
        lon, lat = cross_track_point(t[min_index, 0], t[min_index, 1], t[min_index + 1, 0], t[min_index + 1, 1],
                                     pt[0], pt[1])
    return lon, lat, min_index


