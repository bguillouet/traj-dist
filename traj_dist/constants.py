
from enum import Enum


class EarthRadius(Enum):
    """
    earth radius in meters
    """
    MEAN = 0
    EQUATORIAL = 1


earth_radius_type = EarthRadius.MEAN


def earth_radius():
    if earth_radius_type == EarthRadius.MEAN:
        return 6371008.8
    elif earth_radius_type == EarthRadius.EQUATORIAL:
        return 6378137.0
    return EarthRadius.MEAN
