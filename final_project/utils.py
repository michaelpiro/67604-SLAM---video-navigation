import os
import cv2


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