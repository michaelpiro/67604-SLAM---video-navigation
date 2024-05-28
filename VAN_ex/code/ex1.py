import cv2
import random
import matplotlib.pyplot as plt
import os

NUM_FEATURES_TO_SHOW = 20
MAX_FEATURES = 501
BAD_RATIO = 1
GOOD_RATIO = 0.6
MAC = True
if MAC:
    DATA_PATH = 'VAN_ex/dataset/sequences/00'
else:
    DATA_PATH = r'...\VAN_ex\dataset\sequences\00\\'


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
    # img1 = cv2.imread(DATA_PATH+'image_0\\'+img_name, 0)
    # img2 = cv2.imread(DATA_PATH+'image_1\\'+img_name, 0)
    return img1, img2

def q1_1(img_left, img_right, kp_left, kp_right):
    """
    draw keypoints on images
    :param img_left: left image
    :param img_right: right image
    :param kp_left: keypoint for left image
    :param kp_right: keypoints for right image
    :return:    None
    """
    kp_left_image = cv2.drawKeypoints(image=img_left,
                                      keypoints=kp_left,
                                      outImage=None,
                                      color=(0, 255, 0),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    kp_right_image = cv2.drawKeypoints(image=img_right,
                                       keypoints=kp_right,
                                       outImage=None,
                                       color=(0, 255, 0),
                                       flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    """ show images with keypoints (question 1.1)"""
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(kp_left_image)
    plt.title('Left Image Key Points')

    plt.subplot(2, 1, 2)
    plt.imshow(kp_right_image)
    plt.title('Right Image Key Points')

def q1_2(desc_left):
    """
    print the descriptors of the first two features in the left image
    :param desc_left:   descriptors of the left image
    :return:            None
    """
    print(f"Descriptor of the first feature in the left image:\n {desc_left[0]}")
    print(f"Descriptor of the second feature in the left image:\n {desc_left[1]}")

def q1_3(matches_knn2, img_left, img_right, kp_left, kp_right):
    """
    match the key points and draw the matches
    :param matches_knn2:                knn matches
    :param img_left:                    left image
    :param img_right:                   right image
    :param kp_left:                     keypoints of the left image
    :param kp_right:                    keypoints of the right image
    :return:                            None
    """
    """creating masks for drawing matches"""
    total_matches = len(matches_knn2)
    mask_for_rand = [[0, 0]] * total_matches
    random_matches_idx = random.sample(range(len(mask_for_rand)), NUM_FEATURES_TO_SHOW)
    for i in random_matches_idx:
        mask_for_rand[i] = [1, 0]
    """draw the random 20 matches without ratio test"""
    plt.figure()
    plt.subplot(2, 1, 1)
    rand_20_img = cv2.drawMatchesKnn(img_left, kp_left, img_right, kp_right, matches_knn2,
                                     outImg=None,
                                     matchColor=(255, 0, 0),
                                     singlePointColor=(0, 255, 0),
                                     matchesMask=mask_for_rand,
                                     flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(rand_20_img)
    plt.title('20 Random matches without ratio test')

def q1_4(matches_knn2, img_left, img_right, kp_left, kp_right):
    """
    apply ratio test to find bad matches (for question 1.4),
    :param matches_knn2:    knn matches
    :param img_left:        left image
    :param img_right:       right image
    :param kp_left:         keypoints of the left image
    :param kp_right:        keypoints of the right image
    :return:                None
    """
    """
    apply ratio test to find bad matches (for question 1.4), 
    keep the indices of bad matches in failed_indices
    and use it afterwards to create masks and show the matches that discarded
    """
    total_matches = len(matches_knn2)
    good_matches = []
    bad_matches = []
    for i, (m, n) in enumerate(matches_knn2):
        if m.distance < GOOD_RATIO * n.distance:
            good_matches.append([m])
        else:
            bad_matches.append([m])
    mask_for_ratio_test = [[0, 0]] * len(good_matches)
    mask_for_discarded = [[0, 0]] * len(bad_matches)
    """
    generate random indices to show some of the matches in the image, then check if the match is good or bad
    and use it to create masks for drawing matches
    """
    random_good_idx = random.sample(range(len(good_matches)), NUM_FEATURES_TO_SHOW)
    random_bad_idx = random.sample(range(len(bad_matches)), 5)


    for i in random_good_idx:
        mask_for_ratio_test[i] = [1, 0]
    for i in random_bad_idx:
        mask_for_discarded[i] = [1, 0]


    """
    compute and print the number of good matches, discarded matches and total matches
    """
    # discarded_matches_count = sum(failed_indices)
    discarded_matches_count = total_matches - len(good_matches)
    good_matches_count = total_matches - discarded_matches_count
    print(f"Ratio value used: {GOOD_RATIO}")
    print(f"Total matches: {total_matches}")
    print(f"Good matches: {good_matches_count}")
    print(f"Discarded matches: {discarded_matches_count}")


    """draw 20 random matches from the matches that passed the ratio test (question 1.4)"""
    passed_ratio_t_img = cv2.drawMatchesKnn(img_left, kp_left, img_right, kp_right, good_matches,
                                            outImg=None,
                                            matchColor=(255, 0, 0),
                                            singlePointColor=(0, 255, 0),
                                            matchesMask=mask_for_ratio_test,
                                            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    plt.subplot(2, 1, 2)
    plt.imshow(passed_ratio_t_img)
    plt.title('The Matches that passed the ratio test')
    plt.show()

    """ show a match that is discarded by the ratio test but is actually a good match"""
    plt.figure()
    failed_ratio_t_img = cv2.drawMatchesKnn(img_left, kp_left, img_right, kp_right,[bad_matches[243]],
                                                outImg=None,
                                                matchColor=(255, 0, 0),
                                                singlePointColor=(0, 255, 0),
                                                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(failed_ratio_t_img)
    plt.title('good match that discarded by the ratio test')
    plt.show()


def q1(idx):
    feature = cv2.ORB_create(MAX_FEATURES)
    img_left, img_right = read_images(idx)
    kp_left, desc_left = feature.detectAndCompute(img_left, None)
    kp_right, desc_right = feature.detectAndCompute(img_right, None)


    """draw keypoints on images - Question 1.1"""
    q1_1(img_left, img_right, kp_left, kp_right)

    """print the descriptors of the first two features in the left image - Question 1.2"""
    q1_2(desc_left)

    """ match the key points and draw the matches (question 1.3)"""
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_knn2 = matcher.knnMatch(desc_left, desc_right, k=2)
    q1_3(matches_knn2,img_left, img_right, kp_left, kp_right)

    """ratio test - question 1.4"""
    q1_4(matches_knn2, img_left, img_right, kp_left, kp_right)

if __name__ == '__main__':
    q1(0)
