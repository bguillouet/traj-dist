import math
import numpy as np

rad = math.pi / 180.0
R = 6378137.0


def great_circle_distance(lon1, lat1, lon2, lat2):
    """
    Usage
    -----
    Compute the great circle distance, in meter, between (lon1,lat1) and (lon2,lat2)

    Parameters
    ----------
    param lat1: float, latitude of the first point
    param lon1: float, longitude of the first point
    param lat2: float, latitude of the se*cond point
    param lon2: float, longitude of the second point

    Returns
    -------x
    d: float
       Great circle distance between (lon1,lat1) and (lon2,lat2)
    """

    dlat = rad * (lat2 - lat1)
    dlon = rad * (lon2 - lon1)
    a = (math.sin(dlat / 2.0) * math.sin(dlat / 2.0) +
         math.cos(rad * lat1) * math.cos(rad * lat2) *
         math.sin(dlon / 2.0) * math.sin(dlon / 2.0))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d


def great_circle_distance_traj(lons1, lats1, lons2, lats2, l1, l2):
    """
    Usage
    -----
    Compute pairwise great circle distance, in meter, between longitude/latitudes coordinates.

    Parameters
    ----------
    param lats1: float, latitudes of the firs trajectories
    param lons1: float, longitude of the trajectories
    param lats2: float, latitudes of the se*cond trajectories
    param lons2: float, longitudess of the second trajectories
    param l1 : int, length of the first trajectories
    param l2 : int, length of the second trajectories

    Returns
    -------x
    d: float
       Great circle distance between (lon1,lat1) and (lon2,lat2)
    """

    mdist = np.empty((l1, l2), dtype=float)
    for i in range(l1):
        for j in range(l2):
            mdist[i, j] = great_circle_distance(lons1[i], lats1[i], lons2[j], lats2[j])
    return mdist


def initial_bearing(lon1, lat1, lon2, lat2):
    """
    Usage
    -----
    Bearing between (lon1,lat1) and (lon2,lat2), in degree.

    Parameters
    ----------
    param lat1: float, latitude of the first point
    param lon1: float, longitude of the first point
    param lat2: float, latitude of the second point
    param lon2: float, longitude of the second point

    Returns
    -------
    brng: float
           Bearing between (lon1,lat1) and (lon2,lat2), in degree.

    """
    dlon = rad * (lon2 - lon1)
    y = math.sin(dlon) * math.cos(rad * (lat2))
    x = math.cos(rad * (lat1)) * math.sin(rad * (lat2)) - math.sin(rad * (lat1)) * math.cos(rad * (lat2)) * math.cos(
        dlon)
    ibrng = math.atan2(y, x)

    return ibrng


def cross_track_distance(lon1, lat1, lon2, lat2, lon3, lat3, d13):
    """
    Usage
    -----
    Angular cross-track-distance of a point (lon3, lat3) from a great-circle path between (lon1, lat1) and (lon2, lat2)
    The sign of this distance tells which side of the path the third point is on.

    Parameters :
    ----------
    param lat1: float, latitude of the first point
    param lon1: float, longitude of the first point
    param lat2: float, latitude of the second point
    param lon2: float, longitude of the second point
     param lat3: float, latitude of the third point
    param lon3: float, longitude of the third point

    Usage
    -----
    crt: float
         the (angular) cross_track_distance

    """

    theta13 = initial_bearing(lon1, lat1, lon3, lat3)  # bearing from start point to third point
    theta12 = initial_bearing(lon1, lat1, lon2, lat2)  # bearing from start point to end point

    crt = math.asin(math.sin(d13 / R) * math.sin(theta13 - theta12)) * R

    return crt


def along_track_distance(crt, d13):
    """
    Usage
    -----
    The along-track distance from the start point (lon1, lat1) to the closest point on the the path
    to the third point (lon3, lat3).

    Parameters
    ----------
    param crt : float, cross_track_distance
    param d13 : float, along_track_distance

    Returns
    -------
    alt: float
         The along-track distance
    """

    alt = math.acos(math.cos(d13 / R) / math.cos(crt / R)) * R
    return alt


def point_to_path(lon1, lat1, lon2, lat2, lon3, lat3, d13, d23, d12):
    """
    Usage
    -----
    The point-to-path distance between point (lon3, lat3) and path delimited by (lon1, lat1) and (lon2, lat2).
    The point-to-path distance is the cross_track distance between the great circle path if the projection of
    the third point lies on the path. If it is not on the path, return the minimum of the
    great_circle_distance between the first and the third or the second and the third point.

    Parameters
    ----------
    param lat1: float, latitude of the first point
    param lon1: float, longitude of the first point
    param lat2: float, latitude of the second point
    param lon2: float, longitude of the second point
    param lat3: float, latitude of the third point
    param lon3: float, longitude of the third point
    param d13 : float, great circle distance between (lon1, lat1) and (lon3, lat3)
    param d23 : float, great circle distance between (lon2, lat2) and (lon3, lat3)
    param d12 : float, great circle distance between (lon1, lat1) and (lon2, lat2)


    Returns
    -------

    ptp : float
          The point-to-path distance between point (lon3, lat3) and path delimited by (lon1, lat1) and (lon2, lat2)

    """
    crt = cross_track_distance(lon1, lat1, lon2, lat2, lon3, lat3, d13)
    d1p = along_track_distance(crt, d13)
    d2p = along_track_distance(crt, d23)
    if (d1p > d12) or (d2p > d12):
        ptp = np.min((d13, d23))
    else:
        ptp = np.abs(crt)
    return ptp
