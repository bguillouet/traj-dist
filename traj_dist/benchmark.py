import pickle
import traj_dist.distance as tdist
import timeit
import collections
import pandas as pd

traj_list = pickle.load( open("/Users/bguillouet/These/trajectory_distance/data/" + "benchmark_trajectories.pkl","rb"))

time_dict = collections.defaultdict(dict)

for distance in  ["sspd", "frechet", "discret_frechet", "hausdorff", "dtw", "lcss", "edr", "erp"]:
    t_python = timeit.timeit(lambda : tdist.pdist(traj_list[:100], metric=distance, implementation="python"), number=1)
    t_cython = timeit.timeit(lambda : tdist.pdist(traj_list[:100], metric=distance, implementation="cython"), number=1)

    time_dict[distance] = {"Python": t_python, "Cython": t_cython}


df = pd.DataFrame(time_dict)

print(df)

