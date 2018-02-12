import pickle
import traj_dist.distance as tdist
import timeit
import collections
import pandas as pd

traj_list = pickle.load(open("/Users/bguillouet/These/trajectory_distance/data/benchmark_trajectories.pkl", "rb"))

time_dict = collections.defaultdict(dict)

for distance in ["sspd", "frechet", "discret_frechet", "hausdorff", "dtw", "lcss", "edr", "erp", "sowd_grid"]:
#for distance in ["erp"]:
    if not (distance in ["sowd_grid"]):
        t_python = timeit.timeit(lambda: tdist.pdist(traj_list[:100], metric=distance, implementation="python"),
                                 number=1)
        t_cython = timeit.timeit(lambda: tdist.pdist(traj_list[:100], metric=distance, implementation="cython"),
                                 number=1)

    else:
        t_python = -1
        t_cython = -1
    if not (distance in ["frechet", "discret_frechet"]):
        t_python_g = timeit.timeit(
            lambda: tdist.pdist(traj_list[:100], metric=distance, implementation="python", type_d="geographical"),
            number=1)
        t_cython_g = timeit.timeit(
            lambda: tdist.pdist(traj_list[:100], metric=distance, implementation="cython", type_d="geographical"),
            number=1)
    else:
        t_python_g = -1
        t_cython_g = -1

    time_dict[distance] = {"Python": t_python, "Python Geo": t_python_g, "Cython": t_cython, "Cython Geo": t_cython_g}

df = pd.DataFrame(time_dict)

df.to_csv("/Users/bguillouet/These/trajectory_distance/data/benchmark.csv")