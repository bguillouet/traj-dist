import numpy as np
import pandas as pd
import traj_dist.distance as tdist
import timeit


data = pd.read_pickle("/Users/bguillouet/These/trajectory_review/data/extracted/starting_from/Caltrain_city_center_v3.pkl")

traj_list = [group[["lons","lats"]].values for _,group in data.groupby("id_traj")]


print("Start Running")
print(timeit.timeit(lambda : tdist.pdist(traj_list[:100], metric="sspd"), number=1))
print("End Running")

#print("Start Running")
#print(timeit.timeit(lambda : tdist.pdist(traj_list[:1000], metric="sspd", implementation="python"), number=1))
#print("End Running")
