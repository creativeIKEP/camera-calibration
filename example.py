import argparse
import numpy as np
from calibration import calc_camera_param, separate_param


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


def main(csv_filename, skiprows, camera_imagename):
    calibration_data = np.loadtxt(csv_filename, delimiter=',', skiprows=skiprows)
    data_count = calibration_data.shape[0]
    camera_parameter_mat = calc_camera_param(calibration_data, data_count)
    print("camera parameter")
    print(camera_parameter_mat)
    print()

    if camera_imagename:
        reporojection(camera_parameter_mat, calibration_data[:, 0:3], camera_imagename)

    external_param, internal_param = separate_param(camera_parameter_mat)
    print("external parameter")
    print(external_param)
    print()
    print("internal parameter")
    print(internal_param)
    print()

    print("--------------\nusecase example")
    rotation_mat = external_param[0:3, 0:3]
    inv_rotation_mat = np.linalg.inv(rotation_mat)
    transform_vec = external_param[0:3, 3]
    camera_pos = np.matmul(inv_rotation_mat, -transform_vec)
    print("camera position = " + str(camera_pos))


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
