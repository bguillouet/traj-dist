import numpy as np
import traj_dist.distance as tdist
import pickle

traj_list = pickle.load(open("/Users/bguillouet/These/trajectory_distance/data/benchmark_trajectories.pkl", "rb"))[:10]
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
