import pickle
from traj_dist.pydist.sspd import s_pt_to_traj_dist, s_closest_pt

import traj_dist.distance as tdist
import timeit
import collections
import pandas as pd

from traj_dist.pydist.sspd_spherical_vectorized import s_pt_to_traj_dist_vectorized, s_closest_pt_vectorized


def dist_spd_pt(s_spd_pt_func, trajectories):
    for i in range(nb_traj):
        traj_i = trajectories[i]
        for j in range(i + 1, nb_traj):
            traj_j = trajectories[j]

            for pt_i in range(len(traj_i)):
                s_spd_pt_func(traj_i[pt_i], traj_j)


if __name__ == '__main__':
    file_name = f'../data/benchmark_trajectories.pkl'
    file = open(file_name, 'rb')
    traj_list = pickle.load(file, encoding="bytes")
    nb_traj = 100
    traj_list = traj_list[:nb_traj]

    time_dict = collections.defaultdict(dict)

    t_spherical = timeit.timeit(
        lambda: tdist.pdist(traj_list, metric="sspd", type_d="spherical"),  number=1)
    t_p_spherical = timeit.timeit(
        lambda: tdist.pdist(traj_list, metric="sspd", type_d="spherical", impl="p_impl"), number=1)
    t_p_spherical_vectorized = timeit.timeit(
        lambda: tdist.pdist(traj_list, metric="sspd_vectorized", type_d="spherical", impl="p_impl"), number=1)

    t_pt_spherical = timeit.timeit(lambda: dist_spd_pt(s_pt_to_traj_dist, traj_list), number=1)
    t_pt_spherical_vectorized = timeit.timeit(lambda: dist_spd_pt(s_pt_to_traj_dist_vectorized, traj_list), number=1)

    t_pt_closest_spherical = timeit.timeit(lambda: dist_spd_pt(s_closest_pt, traj_list), number=1)
    t_pt_closest_spherical_vectorized = timeit.timeit(lambda: dist_spd_pt(s_closest_pt_vectorized, traj_list), number=1)

    time_dict["sspd"] = {"c-Spherical": t_spherical,
                         "p-Spherical": t_p_spherical,
                         "p-Spherical-Vectorized": t_p_spherical_vectorized}
    time_dict["spd_pt"] = {"c-Spherical": 'na',
                           "p-Spherical": t_pt_spherical,
                           "p-Spherical-Vectorized": t_pt_spherical_vectorized}
    time_dict["spd_closest_pt"] = {"c-Spherical": 'na',
                                   "p-Spherical": t_pt_closest_spherical,
                                   "p-Spherical-Vectorized": t_pt_closest_spherical_vectorized}

    df = pd.DataFrame(time_dict).transpose()
    df.to_csv("../data/benchmark_sspd.csv")

