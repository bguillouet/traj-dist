import Geohash.geohash as geoh
import shapely.geometry as geos
import numpy as np

def linecell_lons_bigger_step(p1, p2, cell_start, lons_all, lats_all, lons_center_all, lats_center_all):
    reverse = False
    if p2[1] < p1[1]:
        tmp = p1
        p1 = p2
        p2 = tmp
        reverse = True

    lats_start_index = np.where(lats_all < p1[1])[0][-1]
    lats_end_index = np.where(lats_all > p2[1])[0][0]
    lats = lats_all[lats_start_index:lats_end_index + 1]

    if p1[0] < p2[0]:
        order = "croissant"
        idx_step = 1
        lons_start_index = np.where(lons_all < p1[0])[0][-1]
        lons_end_index = np.where(lons_all > p2[0])[0][0]
        lons = lons_all[lons_start_index:lons_end_index + 1]
    else:
        order = "decroissant"
        idx_step = -1
        lons_start_index = np.where(lons_all < p2[0])[0][-1]
        lons_end_index = np.where(lons_all > p1[0])[0][0]
        lons = lons_all[lons_start_index:lons_end_index + 1]
        lons = lons[::-1]
    line = geos.LineString([p1, p2])

    nlons = len(lons) - 2
    nlats = len(lats) - 2

    if not (reverse):
        cell = [cell_start]
    else:
        if order == "croissant":
            cell = [[cell_start[0] - nlons, cell_start[1] - nlats]]
        else:
            cell = [[cell_start[0] + nlons, cell_start[1] - nlats]]

    lons_inter = []
    for l in lons[1:-1]:
        lons_shape = geos.LineString([[l, lats[0]], [l, lats[-1]]])
        lons_inter.append(lons_shape.intersection(line))

    idx_lat = 0
    for p_int in lons_inter:
        if p_int.y < lats[idx_lat + 1]:
            cell.append([cell[-1][0] + idx_step, cell[-1][1]])
        else:
            cell.append([cell[-1][0], cell[-1][1] + 1])
            cell.append([cell[-1][0] + idx_step, cell[-1][1]])
            idx_lat += 1
    if p2[1] > lats[idx_lat + 1]:
        cell.append([cell[-1][0], cell[-1][1] + 1])
    if reverse:
        cell.reverse()
    cells_coord = map(lambda x: [lons_center_all[x[0]], lats_center_all[x[1]]], cell)
    return cell, cells_coord


def linecell_lats_bigger_step(p1, p2, cell_start, lons_all, lats_all, lons_center_all, lats_center_all):
    reverse = False
    if p2[0] < p1[0]:
        tmp = p1
        p1 = p2
        p2 = tmp
        reverse = True

    lons_start_index = np.where(lons_all < p1[0])[0][-1]
    lons_end_index = np.where(lons_all > p2[0])[0][0]
    lons = lons_all[lons_start_index:lons_end_index + 1]

    if p1[1] < p2[1]:
        order = "croissant"
        idx_step = 1
        lats_start_index = np.where(lats_all < p1[1])[0][-1]
        lats_end_index = np.where(lats_all > p2[1])[0][0]
        lats = lats_all[lats_start_index:lats_end_index + 1]
    else:
        order = "decroissant"
        idx_step = -1
        lats_start_index = np.where(lats_all < p2[1])[0][-1]
        lats_end_index = np.where(lats_all > p1[1])[0][0]
        lats = lats_all[lats_start_index:lats_end_index + 1]
        lats = lats[::-1]
    line = geos.LineString([p1, p2])

    nlons = len(lons) - 2
    nlats = len(lats) - 2

    if not (reverse):
        cell = [cell_start]
    else:
        if order == "croissant":
            cell = [[cell_start[0] - nlons, cell_start[1] - nlats]]
        else:
            cell = [[cell_start[0] - nlons, cell_start[1] + nlats]]

    lats_inter = []
    for l in lats[1:-1]:
        lats_shape = geos.LineString([[lons[0], l], [lons[-1], l]])
        lats_inter.append(lats_shape.intersection(line))

    idx_lon = 0
    for p_int in lats_inter:
        if p_int.x < lons[idx_lon + 1]:
            cell.append([cell[-1][0], cell[-1][1] + idx_step])
        else:
            cell.append([cell[-1][0] + 1, cell[-1][1]])
            cell.append([cell[-1][0], cell[-1][1] + idx_step])
            idx_lon += 1
    if p2[0] > lons[idx_lon + 1]:
        cell.append([cell[-1][0] + 1, cell[-1][1]])
    if reverse:
        cell.reverse()
    cells_coord = map(lambda x: [lons_center_all[x[0]], lats_center_all[x[1]]], cell)
    return cell, cells_coord


def get_extremum(traj):
    lons = traj[:, 0]
    lats = traj[:, 1]
    min_lon = min(lons)
    min_lat = min(lats)
    max_lon = max(lons)
    max_lat = max(lats)
    return min_lon, min_lat, max_lon, max_lat


def trajectory_set_grid(traj_set, precision, time=False):
    extremums = np.array(map(get_extremum, traj_set))
    p_bottom_left = [min(extremums[:, 0]), min(extremums[:, 1])]
    p_top_right = [max(extremums[:, 2]), max(extremums[:, 3])]
    p_ble = geoh.encode(p_bottom_left[1], p_bottom_left[0], precision)
    p_tre = geoh.encode(p_top_right[1], p_top_right[0], precision)
    lat_ble, lon_ble, dlat, dlon = geoh.decode_exactly(p_ble)
    lat_tre, lon_tre, dlat, dlon = geoh.decode_exactly(p_tre)
    lats_all = np.arange(lat_ble - dlat, lat_tre + (3 * dlat), dlat * 2)
    lons_all = np.arange(lon_ble - dlon, lon_tre + 3 * dlon, dlon * 2)
    lats_center_all = np.arange(lat_ble, lat_tre + 2 * dlat, dlat * 2)
    lons_center_all = np.arange(lon_ble, lon_tre + 2 * dlon, dlon * 2)

    cells_traj = []
    for traj in traj_set:
        p_start = traj[0]
        cell_start_x = np.where(lons_all < p_start[0])[0][-1]
        cell_start_y = np.where(lats_all < p_start[1])[0][-1]
        cell_start = [cell_start_x, cell_start_y]

        cells = []

        for id_seg in range(len(traj) - 1):
            start = traj[id_seg]
            end = traj[id_seg + 1]
            if time:
                cell_start_time = start[2]
            if abs(start[0] - end[0]) / dlon > abs(start[1] - end[1]) / dlat:
                cell, cells_coord = linecell_lons_bigger_step(start, end, cell_start[:2], lons_all, lats_all,
                                                              lons_center_all,
                                                              lats_center_all)
            else:
                cell, cells_coord = linecell_lats_bigger_step(start, end, cell_start[:2], lons_all, lats_all,
                                                              lons_center_all,
                                                              lats_center_all)
            if time:
                if not cells:
                    cell_time = [cell[0] + [True, [cell_start_time]]]
                else:
                    if cell[0] == cells[-1][:2]:
                        cells[-1][3].append(cell_start_time)
                        cell_time = []
                    else:
                        cell_time = [cell[0] + [True, [cell_start_time]]]
                cell_time = cell_time + map(lambda x: x + [False, -1], cell[1:-1])
            else:
                if not cells:
                    cell_time = [cell[0] + [True]]
                else:
                    if cell[0] == cells[-1][:2]:
                        cell_time = []
                    else:
                        cell_time = [cell[0] + [True]]
                cell_time = cell_time + map(lambda x: x + [False], cell[1:-1])

            cells.extend(cell_time)
            cell_start = cell[-1]
        if time:
            cell_end_time = end[2]
            if cell_start == cells[-1][:2]:
                cells[-1][3].append(cell_end_time)
            else:
                cells.append(cell_start + [True, [cell_end_time]])
        else:
            if cell_start != cells[-1][:2]:
                cells.append(cell_start + [True])
        cells_traj.append(cells)
    # cells_traj_=map(np.array,cells_traj)
    return cells_traj, lons_all, lats_all, lons_center_all, lats_center_all


def trajectory_grid(traj_0, precision):
    cells_list, lons_all, lats_all, lons_center_all, lats_center_all = trajectory_set_grid([traj_0], precision)
    return cells_list[0], lons_all, lats_all, lons_center_all, lats_center_all

