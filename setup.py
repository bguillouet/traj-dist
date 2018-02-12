from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("traj_dist.cydist.basic_geographical", ["traj_dist/cydist/basic_geographical.pyx"]),
               Extension("traj_dist.cydist.basic_euclidean", ["traj_dist/cydist/basic_euclidean.pyx"]),
               Extension("traj_dist.cydist.sspd", ["traj_dist/cydist/sspd.pyx"]),
               Extension("traj_dist.cydist.dtw", ["traj_dist/cydist/dtw.pyx"]),
               Extension("traj_dist.cydist.lcss", ["traj_dist/cydist/lcss.pyx"]),
               Extension("traj_dist.cydist.hausdorff", ["traj_dist/cydist/hausdorff.pyx"]),
               Extension("traj_dist.cydist.discret_frechet", ["traj_dist/cydist/discret_frechet.pyx"]),
               Extension("traj_dist.cydist.frechet", ["traj_dist/cydist/frechet.pyx"]),
               Extension("traj_dist.cydist.segment_distance", ["traj_dist/cydist/segment_distance.pyx"]),
               Extension("traj_dist.cydist.sowd", ["traj_dist/cydist/sowd.pyx"]),
               Extension("traj_dist.cydist.erp", ["traj_dist/cydist/erp.pyx"]),
               Extension("traj_dist.cydist.edr", ["traj_dist/cydist/edr.pyx"])]

setup(
    name="trajectory_distance",
    version="1.0",
    author="Brendan Guillouet",
    author_email="brendan.guillouet@gmail.com",
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
    install_requires=["numpy>=1.14.0", "Cython>=0.27.3", "Shapely>=1.6.4", "Geohash==1.0", 'pandas>=0.20.3',
                      'scipy>=0.19.1'],
    description="Distance to compare 2D-trajectories in Cython",
    packages=["traj_dist", "traj_dist.cydist", "traj_dist.pydist"],
)
