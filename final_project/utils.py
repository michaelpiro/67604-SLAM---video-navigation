import os
import pickle
from typing import Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt

from final_project.arguments import *
from final_project.arguments import DATA_PATH, LEN_DATA_SET
from final_project.backend.database.tracking_database import TrackingDB

MAC = True


def rodriguez_to_mat(rvec, tvec):
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))


def read_images(idx):
    """
    read images from the dataset
    :param idx:     index of the image
    :return:    img_left, img_right  images  (left and right)
    """
    img_name = '{:06d}.png'.format(idx)
    path1 = os.path.join(DATA_PATH, 'image_0', img_name)
    path2 = os.path.join(DATA_PATH, 'image_1', img_name)
    img1 = cv2.imread(path1, 0)
    img2 = cv2.imread(path2, 0)
    return img1, img2


def calc_ransac_iteration(inliers_percent):
    suc_prob = 0.9999999999
    outliers_prob = 1 - (inliers_percent / 100) + 0.0000000001
    min_set_size = 4
    ransac_iterations = int(np.log(1 - suc_prob) / np.log(1 - np.power(1 - outliers_prob, min_set_size))) + 1
    return ransac_iterations


def read_cameras():
    if MAC:
        data_path = '/Users/mac/67604-SLAM-video-navigation/VAN_ex/dataset/sequences/00/'
    else:
        data_path = DATA_PATH
    with open(data_path + 'calib.txt') as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2


def read_extrinsic_matrices(file_path=GROUND_TRUTH_PATH, n=LEN_DATA_SET):
    """
    Reads n lines from a file and returns a list of extrinsic camera matrices.

    Args:
    - file_path (str): Path to the file containing the extrinsic matrices.
    - n (int): Number of lines (matrices) to read from the file.

    Returns:
    - List[np.ndarray]: A list of n extrinsic camera matrices (each a 3x4 numpy array).
    """
    extrinsic_matrices = []

    with open(file_path, 'r') as file:
        for i in range(n):
            line = file.readline()
            if not line:
                break  # If less than n lines are available, stop reading
            numbers = list(map(float, line.strip().split()))
            if len(numbers) != 12:
                raise ValueError(f"Line {i + 1} does not contain 12 numbers.")
            matrix = np.array(numbers).reshape(3, 4)
            extrinsic_matrices.append(matrix)

    return extrinsic_matrices


def get_ground_truth_locations():
    all_transformations = read_extrinsic_matrices()
    cameras_locations2 = []
    for cam in all_transformations:
        rot = cam[:3, :3]
        t = cam[:3, 3]
        cameras_locations2.append(-rot.T @ t)
    return cameras_locations2


def visualize_track(tracking_db: TrackingDB, trackId: int):
    def get_feature_location(tracking_db: TrackingDB, frameId: int, trackId: int) -> Tuple[float, float]:
        link = tracking_db.linkId_to_link[(frameId, trackId)]
        return link.x_left, link.y

    frames = tracking_db.frames(trackId)
    print(f"Track {trackId} has {len(frames)} frames")
    plt.figure()
    num_frames = min(len(frames), 8)
    for i in range(0, num_frames, 1):
        # print(f"Frame {frames[i]}")
        frameId = frames[i]
        img, _ = read_images(frameId)
        x_left, y = get_feature_location(tracking_db, frameId, trackId)
        x_min = int(max(x_left - 10, 0))
        x_max = int(min(x_left + 10, img.shape[1]))
        y_min = int(max(y - 10, 0))
        y_max = int(min(y + 10, img.shape[0]))
        cutout = img[y_min:y_max, x_min:x_max]

        plt.subplot(num_frames, 2, 2 * i + 1)
        plt.imshow(img, cmap='gray')
        plt.scatter(x_left, y, color='red')  # Center of the cutout

        plt.subplot(num_frames, 2, 2 * i + 2)
        plt.imshow(cutout, cmap='gray')
        plt.scatter([10], [10], color='red', marker='x', linewidths=1)  # Center of the cutout
        if i == 0:
            plt.title(f"Frame {frameId}, Track {trackId}")
    plt.show()


def load(base_filename):
    """
    Load serialized data from a pickle file.

    :param base_filename: Base filename without extension.
    :return: Loaded data object.
    """
    filename = base_filename + '.pkl'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print('Bundles loaded from', filename)
    return data


K, M1, M2 = read_cameras()
P, Q = K @ M1, K @ M2
