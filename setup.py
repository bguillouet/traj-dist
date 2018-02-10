from setuptools import setup

from traj_dist.pydist.sspd import cc_sspd



setup(
    name = "trajectory_distance",
    version = "1.0",
    author = "Brendan Guillouet",
    author_email = "brendan.guillouet@gmail.com",
    ext_modules=[cc_sspd.distutils_extension()],
    install_requires =  ["numpy>=1.9.1", "shapely>=1.5.6", "Geohash"],
    description = "Distance to compare trajectories in Numba",
    packages = ["traj_dist", "traj_dist.pydist"],
)
