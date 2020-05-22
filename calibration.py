import argparse
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


def reporojection(camera_parameter_mat, world_points, camera_imagename):
    data_count = world_points.shape[0]
    reprojection_points = np.empty((data_count, 2))
    for i in range(data_count):
        p = np.append(world_points[i], 1)
        homogeneous_point = np.matmul(camera_parameter_mat, p)
        reprojection_points[i, 0] = homogeneous_point[0] / homogeneous_point[2]
        reprojection_points[i, 1] = homogeneous_point[1] / homogeneous_point[2]

    im = Image.open(camera_imagename)
    im_list = np.asarray(im)
    fig, ax = plt.subplots()
    ax.scatter(reprojection_points[:,0], reprojection_points[:,1], s=5)
    plt.imshow(im_list)
    plt.show()

def separate_param(camera_parameter_mat):
    internal_param = np.zeros((3, 3))
    external_param = np.zeros((4, 4))
    internal_param[2][2] = 1
    external_param[3][3] = 1

    cp31 = camera_parameter_mat[2][0]
    cp32 = camera_parameter_mat[2][1]
    cp33 = camera_parameter_mat[2][2]
    scale = 1.0 / math.sqrt(cp31*cp31 + cp32*cp32 + cp33*cp33)
    camera_parameter_mat *= scale
    external_param[2] = camera_parameter_mat[2]

    internal_param[0][2] = np.dot(camera_parameter_mat[0][0:3], camera_parameter_mat[2][0:3])
    internal_param[1][2] = np.dot(camera_parameter_mat[1][0:3], camera_parameter_mat[2][0:3])

    c1c3_cross = np.cross(camera_parameter_mat[0][0:3], camera_parameter_mat[2][0:3])
    c2c3_cross = np.cross(camera_parameter_mat[1][0:3], camera_parameter_mat[2][0:3])
    cos = -np.dot(c1c3_cross, c2c3_cross) / (np.linalg.norm(c1c3_cross) * np.linalg.norm(c2c3_cross))
    sin = math.sqrt(1 - cos*cos)
    cot = cos / sin
    internal_param[0][0] = np.linalg.norm(c1c3_cross) * sin
    internal_param[0][1] = -internal_param[0][0] * cot
    internal_param[1][1] = np.linalg.norm(c2c3_cross)

    external_param[1][0:3] = internal_param[1][1] * (camera_parameter_mat[1][0:3] - internal_param[1][2] * external_param[2][0:3])
    external_param[1][3] = internal_param[1][1] * (camera_parameter_mat[1][3] - internal_param[1][2] * external_param[2][3])

    external_param[0][0:3] = (camera_parameter_mat[0][0:3] + internal_param[0][0]*cot*external_param[1][0:3] - internal_param[0][2] * external_param[2][0:3]) / internal_param[0][0]
    external_param[0][3] = (camera_parameter_mat[0][3] + internal_param[0][0]*cot*external_param[1][3] - internal_param[0][2] * external_param[2][3]) / internal_param[0][0]

    return external_param, internal_param

def main(csv_filename, skiprows, camera_imagename):
    calibration_data = np.loadtxt(csv_filename, delimiter=',', skiprows=skiprows)
    data_count = calibration_data.shape[0]
    camera_parameter_mat = calc_camera_param(calibration_data, data_count)
    print("camera parameter")
    print(camera_parameter_mat)

    if camera_imagename:
        reporojection(camera_parameter_mat, calibration_data[:, 0:3], camera_imagename)

    external_param, internal_param = separate_param(camera_parameter_mat)
    print(external_param)
    print(internal_param)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help=
        'Input CSV file path. Row is one point data.\
        1st column is x coordinate of real world 3D point.\
        2nd column is y coordinate of real world 3D point.\
        3rd column is z coordinate of real world 3D point.\
        4th column is x coordinate on the camera image.\
        5th column is y coordinate on the camera image.')
    parser.add_argument('-skiprows', type=int, default=1, help=
        "Default is 1.\
        Number of header rows in input csv file.\
        This number of first rows is excluded from the data.")
    parser.add_argument('-image', help=
        "Camera image path.\
        Show reporojection points image if this set.")
    args = parser.parse_args()
    main(args.file, args.skiprows, args.image)
