import pickle
import traj_dist.distance as tdist
import collections
import pandas as pd

traj_list = pickle.load(open("/Users/bguillouet/These/trajectory_distance/data/benchmark_trajectories.pkl", "rb"))


time_dict = collections.defaultdict(dict)

for distance in ["sspd","dtw", "frechet", "discret_frechet", "hausdorff", "dtw", "lcss", "edr", "erp", "sowd_grid"]:
    if not (distance in ["sowd_grid"]):
        c_python= tdist.pdist(traj_list[:100], metric=distance, implementation="python")
        c_cython = tdist.pdist(traj_list[:100], metric=distance, implementation="cython")
        diff_eucl = max(c_python - c_cython)
    else:
        diff_eucl = -1
    if not (distance in ["frechet", "discret_frechet"]):
        c_python_g = tdist.pdist(traj_list[:100], metric=distance, implementation="python", type_d="geographical")
        c_cython_g = tdist.pdist(traj_list[:100], metric=distance, implementation="cython", type_d="geographical")
        diff_geo = max(c_python_g - c_cython_g)
    else:
        diff_geo = -1

    time_dict[distance] = {"eucl": diff_eucl, "Geo" : diff_geo}
    print(time_dict)

df = pd.DataFrame(time_dict)

df.to_csv("/Users/bguillouet/These/trajectory_distance/data/validation.csv")