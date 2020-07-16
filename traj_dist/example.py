import numpy as np
import traj_dist.distance as tdist
import pickle

#because pickle compatibility problem between python 3.6 and python 2.7,we should use this way to open old pkl file
#refer https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3?newreg=8c4b4c32700e4119a3735ecd1f0bb5ca  
with open('/Users/bguillouet/These/trajectory_distance/data/benchmark_trajectories.pkl', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    traj_list = u.load()[:10]
traj_A = traj_list[0]
traj_B = traj_list[1]



# Simple distance

dist = tdist.sspd(traj_A, traj_B)
print(dist)

# Pairwise distance

pdist = tdist.pdist(traj_list, metric="sspd")
print(pdist)

# Distance between two list of trajectories

cdist = tdist.cdist(traj_list, traj_list, metric="sspd")
print(cdist)
