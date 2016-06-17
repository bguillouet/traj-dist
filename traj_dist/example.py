import numpy as np

# Three 2-D Trajectory
traj_A = np.array([[-122.39534, 37.77678],[-122.3992 , 37.77631],[-122.40235, 37.77594],[-122.40553, 37.77848],
                   [-122.40801, 37.78043],[-122.40837, 37.78066],[-122.41103, 37.78463],[-122.41207, 37.78954],
                   [-122.41252, 37.79232],[-122.41316, 37.7951 ],[-122.41392, 37.7989 ],[-122.41435, 37.80129],
                   [-122.41434, 37.80129]])
traj_B = np.array([[-122.39472, 37.77672],[-122.3946 , 37.77679],[-122.39314, 37.77846],[-122.39566, 37.78113],
                   [-122.39978, 37.78438],[-122.40301, 37.78708],[-122.4048 , 37.78666],[-122.40584, 37.78564],
                   [-122.40826, 37.78385],[-122.41061, 37.78321],[-122.41252, 37.78299]])
traj_C = np.array([[-122.39542, 37.77665],[-122.3988 , 37.77417],[-122.41042, 37.76944],[-122.41459, 37.77016],
                   [-122.41462, 37.77013]])
traj_list = [traj_A, traj_B, traj_C]

import traj_dist.distance as tdist

# Simple distance

dist = tdist.sspd(traj_A,traj_B)
print(dist)

# Pairwise distance

pdist = tdist.pdist(traj_list,metric="sspd")
print(pdist)

# Distance between two list of trajectories

cdist = tdist.cdist(traj_list, traj_list,metric="sspd")
print(cdist)
