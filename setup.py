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
    name="traj_dist",
    version="1.2",
    license="MIT",
    author="Brendan Guillouet",
    author_email="brendan.guillouet@gmail.com",
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
    install_requires=["numpy>=1.23.5", "Cython>=0.27.3", "Shapely>=1.8.5.post1", "geohash2==1.1", "pandas>=1.5.1",
                      "scipy>=1.9.3", "scikit-learn>=1.1.3"],
    description="Distance to compare 2D-trajectories in Python/Cython",
    download_url='https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
    keywords=['trajectory', "distance", "haversine"],
    packages=["traj_dist", "traj_dist.pydist"],
    url='https://github.com/bguillouet/traj-dist',
    download_urt='https://github.com/bguillouet/traj-dist/archive/1.1.tar.gz'
)
