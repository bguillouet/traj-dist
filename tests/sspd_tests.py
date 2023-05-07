import pickle
import unittest

import geopy as geopy
import geopy.distance as geopy_distance
import numpy as np
import pandas as pd

from traj_dist.cydist.sspd import c_g_sspd

from traj_dist.pydist.basic_spherical import great_circle_distance
from traj_dist.pydist.sspd import s_sspd, s_pt_to_traj_dist, s_closest_pt
from traj_dist.pydist.sspd_spherical_vectorized import s_sspd_vectorized, s_pt_to_traj_dist_vectorized, \
    s_closest_pt_vectorized, s_spd_vectorized_distances, s_pt_to_traj_dist_threshold_vectorized
from traj_dist.trajectory_utils import remove_duplicate_adjacent_pts

results_file_name = f'../data/benchmark_spherical_sspd_results.csv'
trajectories_file_name = f'../data/benchmark_trajectories.pkl'
columns = ['i', 'j', 'c_dist', 'p_dist']
column_dtypes = {'i': int, 'j': int, 'c_dist': float, 'p_dist': float}
nb_traj = 100


class SphericalSSPDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # load test data
        with open(trajectories_file_name, 'rb') as file:
            traj_list = pickle.load(file, encoding="bytes")
            cls.traj_list = traj_list[:nb_traj]

        cls.df_expected_results = pd.read_csv(results_file_name, dtype=column_dtypes, index_col=0)
        cls.max_diff = max(abs(cls.df_expected_results['c_dist'] - cls.df_expected_results['p_dist']))

    def test_init(self):
        print(self.max_diff)
        for _, row in self.df_expected_results.iterrows():
            i = int(row["i"])
            j = int(row["j"])
            p_dist = s_sspd(self.traj_list[i], self.traj_list[j])
            expected_res = row["p_dist"]
            self.assertAlmostEqual(p_dist, expected_res)

    def test_sspd_vectorized(self):
        for _, row in self.df_expected_results.iterrows():
            i = int(row["i"])
            j = int(row["j"])
            traj_i = self.traj_list[i]
            traj_j = self.traj_list[j]
            p_dist_vectorized = s_sspd_vectorized(traj_i, traj_j)
            expected_res = row["p_dist"]  # saved results
            self.assertAlmostEqual(p_dist_vectorized, expected_res)

    def test_spd_vectorized_distances(self):
        for _, row in self.df_expected_results.iterrows():
            i = int(row["i"])
            j = int(row["j"])
            traj_i = self.traj_list[i]
            traj_j = self.traj_list[j]
            p_dist_vectorized_1, *rest = s_spd_vectorized_distances(traj_i, traj_j)
            p_dist_vectorized_2, *rest = s_spd_vectorized_distances(traj_j, traj_i)
            expected_res = row["p_dist"]  # saved results
            self.assertAlmostEqual(p_dist_vectorized_1 + p_dist_vectorized_2, expected_res)

    def test_sspd_vectorized_remove_duplicate_pts(self):
        for _, row in self.df_expected_results.iterrows():
            i = int(row["i"])
            j = int(row["j"])
            traj_i = remove_duplicate_adjacent_pts(self.traj_list[i])
            traj_j = remove_duplicate_adjacent_pts(self.traj_list[j])
            p_dist_vectorized = s_sspd_vectorized(traj_i, traj_j)
            expected_dist = s_sspd(traj_i, traj_j)
            self.assertAlmostEqual(p_dist_vectorized, expected_dist)

    def test_s_pt_to_traj_dist_vectorized(self):
        for _, row in self.df_expected_results.iterrows():
            i = int(row["i"])
            j = int(row["j"])
            traj_i = self.traj_list[i]
            traj_j = self.traj_list[j]

            for pt_i in range(len(traj_i)):
                expected_res = s_pt_to_traj_dist(traj_i[pt_i], traj_j)
                p_dist_vectorized = s_pt_to_traj_dist_vectorized(traj_i[pt_i], traj_j)
                self.assertAlmostEqual(p_dist_vectorized, expected_res)

    def test_s_pt_to_traj_dist_correctness(self):
        """
        Creates trajectories 0f 2 points on the same latitude, 1 km apart from one another
        and a third point, between the 2 points and 0.5 km south
        Test that the distance from the 3rd point to the trajectory is approx 0.5 km
        Compare the s_pt_to_traj_dist_vectorized distance to great_circle_distance (should be almost equal)
        """
        for lat in range(-50, 50, 2):
            for long in range(-120, 120, 10):
                start = geopy.Point(latitude=lat, longitude=long)
                # Define a general distance objects, initialized with a distance of 1 km and 0.5 km.
                d1 = geopy_distance.distance(meters=1000)
                d2 = geopy_distance.distance(meters=500)
                traj_pt = d1.destination(point=start, bearing=90)  # east
                pt_on_traj = d2.destination(point=start, bearing=90)  # east
                pt = d2.destination(point=pt_on_traj, bearing=180)  # south
                traj_i = np.array([[long, lat], [traj_pt.longitude, traj_pt.latitude]])
                dist = s_pt_to_traj_dist_vectorized(np.array([pt.longitude, pt.latitude]), traj_i)
                dist_expected = great_circle_distance(pt.longitude, pt.latitude,
                                                      pt_on_traj.longitude, pt_on_traj.latitude)
                self.assertAlmostEqual(dist, dist_expected, places=2)  # centimeters' accuracy

    def test_s_closest_pt_to_traj_vectorized(self):
        for _, row in self.df_expected_results.iterrows():
            i = int(row["i"])
            j = int(row["j"])
            traj_i = self.traj_list[i]
            traj_j = self.traj_list[j]

            for pt_i in range(len(traj_i)):
                lon1, lat1, idx1 = s_closest_pt(traj_i[pt_i], traj_j)
                lon2, lat2, idx2 = s_closest_pt_vectorized(traj_i[pt_i], traj_j)

                self.assertAlmostEqual(lon1, lon2)
                self.assertAlmostEqual(lat1, lat2)
                self.assertEqual(idx1, idx2)

    def test_s_pt_to_traj_dist_threshold_vectorized(self):
        threshold_dist = 500

        for _, row in self.df_expected_results.iterrows():
            i = int(row["i"])
            j = int(row["j"])
            traj_i = self.traj_list[i]
            traj_j = self.traj_list[j]

            for pt_i in range(len(traj_i)):
                expected_res = s_pt_to_traj_dist(traj_i[pt_i], traj_j)
                dist, idx, pt_idx = s_pt_to_traj_dist_threshold_vectorized(threshold_dist, traj_i[pt_i], traj_j)
                if idx is None:
                    self.assertGreaterEqual(expected_res, threshold_dist)
                    self.assertGreaterEqual(dist, threshold_dist)
                else:
                    self.assertLessEqual(idx, len(traj_j) - 1)
                    self.assertGreaterEqual(idx, 0)
                    self.assertLessEqual(dist, threshold_dist)


def create_test_data():
    """Computes distance between each pair of the two list of trajectories"""
    file = open(trajectories_file_name, 'rb')
    traj_list = pickle.load(file, encoding="bytes")
    traj_list = traj_list[:nb_traj]
    # create pairwise distance tests
    # nb_traj_pairs = sum(range(nb_traj))
    results = []
    for i in range(nb_traj):
        traj_list_i = traj_list[i]
        for j in range(i + 1, nb_traj):
            traj_list_j = traj_list[j]
            c_dist = c_g_sspd(traj_list_i, traj_list_j)
            p_dist = s_sspd(traj_list_i, traj_list_j)
            results.append([i, j, c_dist, p_dist])
    results_df = pd.DataFrame(results, columns=columns)
    max_diff = max(abs(results_df['c_dist']-results_df['p_dist']))
    print(f'precision difference: {max_diff}')
    # results_df.to_csv(results_file_name)


if __name__ == '__main__':
    # create_test_data()
    unittest.main()
