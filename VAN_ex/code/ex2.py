import cv2
import random
import matplotlib.pyplot as plt
import os
import numpy as np

MAC = True
if MAC:
    DATA_PATH = 'VAN_ex/dataset/sequences/00'
    # DATA_PATH = 'dataset/sequences/00'
else:
    DATA_PATH = r'...\VAN_ex\dataset\sequences\00\\'

NUM_FEATURES_TO_SHOW = 20
MAX_FEATURES = 501
BAD_RATIO = 1
GOOD_RATIO = 0.6
IDX = 0


def read_images(idx):
    """
    read images from the dataset
    :param idx:     index of the image
    :return:    img1, img2  images  (left and right)
    """
    img_name = '{:06d}.png'.format(idx)
    path1 = os.path.join(DATA_PATH, 'image_0', img_name)
    path2 = os.path.join(DATA_PATH, 'image_1', img_name)
    img1 = cv2.imread(path1, 0)
    img2 = cv2.imread(path2, 0)
    print(img1.shape)
    # img1 = cv2.imread(DATA_PATH+'image_0\\'+img_name, 0)
    # img2 = cv2.imread(DATA_PATH+'image_1\\'+img_name, 0)
    return img1, img2


feature = cv2.ORB_create(MAX_FEATURES)
img_left, img_right = read_images(IDX)
kp_left, desc_left = feature.detectAndCompute(img_left, None)
kp_right, desc_right = feature.detectAndCompute(img_right, None)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = matcher.match(desc_left, desc_right)


def calculate_height_dist(matches, kp_left, kp_right):
    """
    calculate the pixel distance and height of the matches
    :param matches: list of matches
    :return:    pixel distance, height
    """
    deviations = []
    for i in range(len(matches)):
        ind_left = matches[i].queryIdx
        ind_right = matches[i].trainIdx
        deviations.append(abs(kp_left[ind_left].pt[1] - kp_right[ind_right].pt[1]))
    return deviations


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


def create_histogram(deviations):
    """
    create histogram of the deviations
    :param deviations: list of deviations
    :return:    None
    """
    plt.hist(deviations, bins=100)
    plt.xlabel('deviations from rectified stereo pattern')
    plt.ylabel('Number of Matches')
    plt.title('Histogram of Pixel Distance')
    plt.show()


def print_precentage_of_deviate_2(deviations):
    """
    print the percentage of the deviations that are greater than 2
    :param deviations: list of deviations
    :return:    None
    """
    count = 0
    for dev in deviations:
        if dev > 2:
            count += 1
    print('Percentage of deviations greater than 2 pixels: ', count / len(deviations) * 100, '%')


def q2_1():
    """
    calculate the pixel distance and height of the matches
    """
    deviations = calculate_height_dist(matches, kp_left, kp_right)
    create_histogram(deviations)
    print_precentage_of_deviate_2(deviations)


def q2_2(matches, kp_left, kp_right):
    """
    present the inliers and outlayers
    :param matches: list of matches
    :return:    None
    """
    inliers = []
    outliers = []
    for i in range(len(matches)):
        ind_left = matches[i].queryIdx
        ind_right = matches[i].trainIdx
        if abs(kp_left[ind_left].pt[1] - kp_right[ind_right].pt[1]) < 2:
            inliers.append(matches[i])
        else:
            outliers.append(matches[i])
    print('Number of inliers: ', len(inliers))
    print('Number of outlayers: ', len(outliers))

    # plt.figure()
    plt.figure(figsize=(15, 8))

    # Concatenate the images
    # space_vector = np.zeros((img_left.shape[0], 1024))
    combined_image = np.vstack((img_left, img_right))
    # combined_image = np.hstack(combined_image, img_right)

    # Show the combined image
    plt.imshow(combined_image, cmap='gray')
    for match in outliers:
        ind_left = match.queryIdx
        ind_right = match.trainIdx
        x1, y1 = kp_left[ind_left].pt
        x2, y2 = kp_right[ind_right].pt
        y2 += img_left.shape[0]   # Adjust x2 for the second image's position
        plt.scatter([x1, x2], [y1, y2], color='cyan', s=0.8)
    # Plot the inliers in orange
    for match in inliers:
        ind_left = match.queryIdx
        ind_right = match.trainIdx
        x1, y1 = kp_left[ind_left].pt
        x2, y2 = kp_right[ind_right].pt
        y2 += img_left.shape[0]   # Adjust x2 for the second image's position
        plt.scatter([x1, x2], [y1, y2], color='orange', s=1)

    plt.axis('off')
    plt.show()
    return inliers, outliers

K, M1, M2 = read_cameras()
P, Q = K @ M1, K @ M2  # multiply by intrinsic camera matrix
def linear_least_squares_triangulation(P, Q, kp_left, kp_right):
    """
    Linear least squares triangulation
    :param P: 2D point in image 1
    :param p1: Camera matrix of image 1
    :param q1: 2D point in image 2
    :param Q: Camera matrix of image 2
    :return:    3D point
    """
    A = np.zeros((4, 4))
    # p_left, p_right = kp_left[inliers[ind].queryIdx], kp_right[inliers[ind].trainIdx]
    p_x, p_y = kp_left
    q_x, q_y = kp_right
    A[0] = P[2] * p_x - P[0]
    A[1] = P[2] * p_y - P[1]
    A[2] = Q[2] * q_x - Q[0]
    A[3] = Q[2] * q_y - Q[1]
    _, _, V = np.linalg.svd(A)
    if V[-1, 3] == 0:
        return V[-1, :3] / (V[-1, 3] + 1e-20)
    return V[-1, :3] / V[-1, 3]

def triangulate_matched_points(P, Q, inliers, kp_left, kp_right):
    """
    Triangulate the matched points
    :param P:
    :param Q:
    :param inliers:
    :param kp_left:
    :param kp_right:
    :return:
    """
    X = np.zeros((len(inliers), 3))
    for i in range(len(inliers)):
        p_left, p_right = kp_left[inliers[i].queryIdx], kp_right[inliers[i].trainIdx]
        # p_x, p_y =
        # q_x, q_y =
        X[i] = linear_least_squares_triangulation(P, Q, p_left.pt, p_right.pt)
    return X

def cv_triangulate_matched_points(inliers):
    """
    Triangulate the matched points using OpenCV
    :param inliers:
    :return:
    """
    X = np.zeros((len(inliers), 3))
    for i in range(len(inliers)):
        p_left, p_right = kp_left[inliers[i].queryIdx], kp_right[inliers[i].trainIdx]
        X_4d = cv2.triangulatePoints(P, Q, p_left.pt, p_right.pt)
        X_4d /= (X_4d[3] + 1e-10)
        X[i] = X_4d[:-1].T
    return X
    # import mpl_toolkits.mplot3d.Axes3D as AX
# mpl_toolkits.mplot3d.Axes3D.scatter

inliers, outliers = q2_2(matches, kp_left, kp_right)
tr = triangulate_matched_points(P, Q, inliers, kp_left, kp_right)
cv_tr = cv_triangulate_matched_points(inliers)
norm = np.linalg.norm(tr - cv_tr, axis=1)
print(np.median(norm))
# print(np.sum(np.square(triangulate_matched_points(P, Q, inliers, kp_left, kp_right) - cv_triangulate_matched_points(inliers))))


# X_4d = cv2.triangulatePoints(P, Q, p_left.pt, p_right.pt)
# X_4d /= (X_4d[3] + 1e-10)
# print(X_4d[:-1].T)


# print(cv_triangulate_matched_points(inliers)[0])
# def q2_3():
