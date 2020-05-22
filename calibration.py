import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math


def calc_camera_param(calibration_data, data_count):
    equation_mat = np.zeros((data_count * 2, 11))
    screen_points_vec = np.zeros(data_count * 2)

    for i in range(data_count):
        p = calibration_data[i]
        equation_mat[i * 2, 0] = p[0]
        equation_mat[i * 2, 1] = p[1]
        equation_mat[i * 2, 2] = p[2]
        equation_mat[i * 2, 3] = 1
        equation_mat[i * 2, 8] = -p[0] * p[3]
        equation_mat[i * 2, 9] = -p[1] * p[3]
        equation_mat[i * 2, 10] = -p[2] * p[3]
        equation_mat[i * 2 + 1, 4] = p[0]
        equation_mat[i * 2 + 1, 5] = p[1]
        equation_mat[i * 2 + 1, 6] = p[2]
        equation_mat[i * 2 + 1, 7] = 1
        equation_mat[i * 2 + 1, 8] = -p[0] * p[4]
        equation_mat[i * 2 + 1, 9] = -p[1] * p[4]
        equation_mat[i * 2 + 1, 10] = -p[2] * p[4]
        screen_points_vec[i * 2] = p[3]
        screen_points_vec[i * 2 + 1] = p[4]

    camera_parameter_vec = np.matmul(equation_mat.T, equation_mat)
    camera_parameter_vec = np.linalg.inv(camera_parameter_vec)
    camera_parameter_vec = np.matmul(camera_parameter_vec, equation_mat.T)
    camera_parameter_vec = np.matmul(camera_parameter_vec, screen_points_vec)
    camera_parameter_vec = np.append(camera_parameter_vec, 1)
    camera_parameter_mat = camera_parameter_vec.reshape(3, 4)
    return camera_parameter_mat


def separate_param(camera_parameter_mat):
    c31 = camera_parameter_mat[2][0]
    c32 = camera_parameter_mat[2][1]
    c33 = camera_parameter_mat[2][2]
    scale = 1.0 / math.sqrt(c31*c31 + c32*c32 + c33*c33)
    camera_parameter_mat *= scale

    r3 = camera_parameter_mat[2][0:3]
    tz = camera_parameter_mat[2][3]
    c1 = camera_parameter_mat[0][0:3]
    c2 = camera_parameter_mat[1][0:3]
    c3 = camera_parameter_mat[2][0:3]

    u0 = np.dot(c1, c3)
    v0 = np.dot(c2, c3)

    c1c3_cross = np.cross(c1, c3)
    c2c3_cross = np.cross(c2, c3)
    cos = -np.dot(c1c3_cross, c2c3_cross) / (np.linalg.norm(c1c3_cross) * np.linalg.norm(c2c3_cross))
    sin = math.sqrt(1 - cos*cos)
    cot = cos / sin
    alfa_u = np.linalg.norm(c1c3_cross) * sin
    alfa_v = np.linalg.norm(c2c3_cross) * sin

    r2 = (c2 - v0 * r3) * sin / alfa_v
    ty = (camera_parameter_mat[1][3] - v0 * tz) * sin / alfa_v

    r1 = (c1 + alfa_u*cot*r2 - u0 * r3) / alfa_u
    tx = (camera_parameter_mat[0][3] + alfa_u*cot*ty - u0 * tz) / alfa_u


    external_param = np.array([
        [r1[0], r1[1], r1[2], tx],
        [r2[0], r2[1], r2[2], ty],
        [r3[0], r3[1], r3[2], tz],
        [0, 0, 0, 1]
    ])
    internal_param = np.array([
        [alfa_u, -alfa_u * cot, u0],
        [0, alfa_v / sin, v0],
        [0, 0, 1]
    ])
    return external_param, internal_param
