import pickle
import traj_dist.distance as tdist
from traj_dist.pydist.linecell import trajectory_set_grid
import timeit
import collections
import pandas as pd
import numpy as np

file_name = f'../data/benchmark_trajectories.pkl'
file = open(file_name, 'rb')
traj_list = pickle.load(file, encoding="bytes")
traj_list = traj_list[:100]

time_dict = collections.defaultdict(dict)

for distance in ["sspd", "frechet", "discret_frechet", "hausdorff", "dtw", "lcss", "edr", "erp"]:
    t_euclidean = timeit.timeit(lambda: tdist.pdist(traj_list, metric=distance), number=1)
    t_p_euclidean = timeit.timeit(lambda: tdist.pdist(traj_list, metric=distance, impl="p_impl"), number=1)

    if not (distance in ["frechet", "discret_frechet"]):
        t_spherical = timeit.timeit(
            lambda: tdist.pdist(traj_list, metric=distance, type_d="spherical"),  number=1)
        t_p_spherical = timeit.timeit(
            lambda: tdist.pdist(traj_list, metric=distance, type_d="spherical", impl="p_impl"), number=1)

    else:
        t_spherical = -1
        t_p_spherical = -1
    time_dict[distance] = {"c-Euclidean": t_euclidean, "c-Spherical": t_spherical,
                           "p-Euclidean": t_p_euclidean, "p-Spherical": t_p_spherical}

t_cells_conversion_dic = collections.defaultdict(int)
for precision in [5, 6, 7]:
    cells_list_, _, _, _, _ = trajectory_set_grid(traj_list, precision=precision)
    cells_list = [np.array(x)[:, :2] for x in cells_list_]

    t_cells_conversion = timeit.timeit(lambda: trajectory_set_grid(traj_list, precision=7), number=1)
    t_cells_conversion_dic[precision] = t_cells_conversion

    t_euclidean = timeit.timeit(
        lambda: tdist.pdist(cells_list, metric="sowd_grid", type_d="euclidean", converted=True), number=1, )
    t_spherical = timeit.timeit(
        lambda: tdist.pdist(cells_list, metric="sowd_grid", type_d="spherical", converted=True), number=1, )
    t_p_euclidean = timeit.timeit(
        lambda: tdist.pdist(cells_list, metric="sowd_grid", type_d="euclidean", impl="p_impl", converted=True), number=1, )
    t_spherical = timeit.timeit(
        lambda: tdist.pdist(cells_list, metric="sowd_grid", type_d="spherical", converted=True), number=1, )
    t_p_spherical = timeit.timeit(
        lambda: tdist.pdist(cells_list, metric="sowd_grid", type_d="spherical", impl="p_impl", converted=True), number=1, )

    time_dict["sowd_grid_%d"%precision] = {"c-Euclidean": t_euclidean, "c-Spherical": t_spherical,
                                           "p-Euclidean": t_p_euclidean, "p-Spherical": t_p_spherical}


df_cells_conversion = pd.Series(t_cells_conversion_dic)
df_cells_conversion.to_csv("../data/benchmark_cells_conversion.csv")

df = pd.DataFrame(time_dict).transpose()
df.to_csv("../data/benchmark.csv")
