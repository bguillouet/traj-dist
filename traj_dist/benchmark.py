import pickle
import traj_dist.distance as tdist
from traj_dist.pydist.linecell import trajectory_set_grid
import timeit
import collections
import pandas as pd
import numpy as np

traj_list = pickle.load(open("/Users/bguillouet/These/trajectory_distance/data/benchmark_trajectories.pkl", "rb"))[:100]

time_dict = collections.defaultdict(dict)

for distance in ["sspd", "frechet", "discret_frechet", "hausdorff", "dtw", "lcss", "edr", "erp"]:
    t_python = timeit.timeit(lambda: tdist.pdist(traj_list, metric=distance, implementation="python"),
                             number=1)
    t_cython = timeit.timeit(lambda: tdist.pdist(traj_list, metric=distance, implementation="cython"),
                             number=1)

    if not (distance in ["frechet", "discret_frechet"]):
        t_python_g = timeit.timeit(
            lambda: tdist.pdist(traj_list, metric=distance, implementation="python", type_d="geographical"),
            number=1)
        t_cython_g = timeit.timeit(
            lambda: tdist.pdist(traj_list, metric=distance, implementation="cython", type_d="geographical"),
            number=1)
    else:
        t_python_g = -1
        t_cython_g = -1
    time_dict[distance] = {"Python": t_python, "Python Geo": t_python_g, "Cython": t_cython, "Cython Geo": t_cython_g}

t_cells_conversion_dic = collections.defaultdict(int)
for precision in [5, 6, 7]:
    cells_list_, _, _, _, _ = trajectory_set_grid(traj_list, precision=precision)
    cells_list = map(lambda x: np.array(x)[:, :2], cells_list_)

    t_cells_conversion = timeit.timeit(lambda: trajectory_set_grid(traj_list, precision=7), number=1)
    t_cells_conversion_dic[precision] = t_cells_conversion

    t_python_g = timeit.timeit(
        lambda: tdist.pdist(cells_list, metric="sowd_grid", implementation="python", type_d="geographical",
                            converted=True), number=1, )
    t_cython_g = timeit.timeit(
        lambda: tdist.pdist(cells_list, metric="sowd_grid", implementation="cython", type_d="geographical",
                            converted=True), number=1)
    time_dict["sowd_grid_%d" % precision] = {"Python": -1, "Python Geo": t_python_g, "Cython": -1,
                                             "Cython Geo": t_cython_g}

df_cells_conversion = pd.Series(t_cells_conversion_dic)
df_cells_conversion.to_csv("/Users/bguillouet/These/trajectory_distance/data/benchmark_cells_conversion.csv")

df = pd.DataFrame(time_dict)
df.to_csv("/Users/bguillouet/These/trajectory_distance/data/benchmark.csv")
