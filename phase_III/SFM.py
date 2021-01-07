
from operator import itemgetter
import numpy as np

def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)

    if abs(tZ) < 10e-6:
        print('tz = ', tZ)

    elif norm_prev_pts.size == 0:
        print('no prev points')

    elif norm_prev_pts.size == 0:
        print('no curr points')

    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data\
            (norm_prev_pts, norm_curr_pts, R, foe, tZ)

    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))

    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []

    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)

        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)

    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    pts_normal = []

    for point in pts:
        pts_normal.append([(point[0] - pp[0]) / focal, (point[1] - pp[1]) / focal])

    return np.array(pts_normal)


def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    pts_unnormal = []

    for point in pts:
        pts_unnormal.append([point[0] * focal + pp[0], point[1] * focal + pp[1]])

    return np.array(pts_unnormal)


def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    R = EM[:3, :3]
    tZ = EM[:3, 3:]
    foe = np.array([tZ[0] / tZ[2], tZ[1] / tZ[2]])

    return R, foe, tZ[2]


def rotate(pts, R):
    # rotate the points - pts using R
    pts_rotated = []

    for p in pts:
        p = np.append(p, np.array(1))
        result = R.dot(p)
        pts_rotated.append([(result[0] / result[2]), (result[1] / result[2])])

    return np.array(pts_rotated)


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index
    x, y = 0, 1
    m = (foe[y]-p[y])/(foe[x]-p[x])
    n = (p[y]*foe[x] - foe[y]*p[x])/(foe[x]-p[x])
    list = [[abs((m * pts[x] + n - pts[y]) / np.sqrt(pow(m, 2) + 1)), i] for i, pts in enumerate(norm_pts_rot)]
    min_dist, i_min = min(list, key=itemgetter(0))

    return i_min, norm_pts_rot[i_min]


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z
    dis_x = tZ * (foe[0] - p_rot[0]) / (p_curr[0] - p_rot[0])
    dis_y = tZ * (foe[1] - p_rot[1]) / (p_curr[1] - p_rot[1])
    dX = abs(foe[0] - p_curr[0])
    dY = abs(foe[1] - p_curr[1])
    ratio = dX / (dY + dX)

    return dis_x * ratio + dis_y * (1 - ratio)