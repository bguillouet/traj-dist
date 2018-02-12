import pandas as pd
import pickle


data = pd.read_pickle("/Users/bguillouet/These/trajectory_review/data/extracted/starting_from/Caltrain_city_center_v3.pkl")

traj_list = [group[["lons","lats"]].values for _,group in data.groupby("id_traj")]

pickle.dump(traj_list, open("/Users/bguillouet/These/trajectory_distance/data/" + "benchmark_trajectories.pkl","wb"))