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
    img_name = '{:06d}.png'.format(idx)
    path1 = os.path.join(DATA_PATH, 'image_0', img_name)
    path2 = os.path.join(DATA_PATH, 'image_1', img_name)
    img1 = cv2.imread(path1, 0)
    img2 = cv2.imread(path2, 0)
    # img1 = cv2.imread(DATA_PATH+'image_0\\'+img_name, 0)
    # img2 = cv2.imread(DATA_PATH+'image_1\\'+img_name, 0)
    return img1, img2


def q1(idx):
    feature = cv2.ORB_create(MAX_FEATURES)
    img_left, img_right = read_images(idx)
    kp_left, desc_left = feature.detectAndCompute(img_left, None)
    kp_right, desc_right = feature.detectAndCompute(img_right, None)

    """draw keypoints on images"""
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

    """ show images """
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(kp_left_image)
    plt.title('Left Image Key Points')

    plt.subplot(2, 1, 2)
    plt.imshow(kp_right_image)
    plt.title('Right Image Key Points')

    """printing the descriptors of the first 2 features"""
    print(f"Descriptor of the first feature in the left image:\n {desc_left[0]}")
    print(f"Descriptor of the second feature in the left image:\n {desc_left[1]}")

    """ match the key points """
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_knn2 = matcher.knnMatch(desc_left, desc_right, k=2)

    """creating masks for drawing matches"""
    total_matches = len(matches_knn2)
    mask_for_rand = [[0, 0]] * total_matches
    mask_for_ratio_test = [[0, 0]] * total_matches
    mask_discarded = [[0, 0]] * total_matches

    """
    apply ratio test to find bad matches, 
    keep the indices of bad matches in failed_indices
    and use it afterwards to create masks and show the matches that discarded
    """
    failed_indices = [0] * total_matches
    for i, (m, n) in enumerate(matches_knn2):
        if m.distance > GOOD_RATIO * n.distance:
            failed_indices[i] = 1

    """
    generate random indices to show some of the matches in the image, then check if the match is good or bad
    and use it to create masks for drawing matches
    """
    random_matches_idx = random.sample(range(len(mask_for_rand)), NUM_FEATURES_TO_SHOW)
    for i in random_matches_idx:
        mask_for_rand[i] = [1, 0]
        if failed_indices[i] == 0:
            mask_for_ratio_test[i] = [1, 0]
        else:
            mask_discarded[i] = [1, 0]
    """
    compute and print the number of good matches, discarded matches and total matches
    """

    discarded_matches_count = sum(failed_indices)
    good_matches_count = total_matches - discarded_matches_count
    print(f"Ratio value used: {GOOD_RATIO}")
    print(f"Total matches: {total_matches}")
    print(f"Good matches: {good_matches_count}")
    print(f"Discarded matches: {discarded_matches_count}")

    """draw the random 20 matches without ratio test"""
    plt.figure()
    plt.subplot(3, 1, 1)
    rand_20_img = cv2.drawMatchesKnn(img_left, kp_left, img_right, kp_right, matches_knn2,
                                     outImg=None,
                                     matchColor=(255, 0, 0),
                                     singlePointColor=(0, 255, 0),
                                     matchesMask=mask_for_rand,
                                     flags=0)
    plt.imshow(rand_20_img)
    plt.title('20 Random matches without ratio test')

    """draw only the matches that passed the ratio test from the random 20 matches"""
    passed_ratio_t_img = cv2.drawMatchesKnn(img_left, kp_left, img_right, kp_right, matches_knn2,
                                            outImg=None,
                                            matchColor=(255, 0, 0),
                                            singlePointColor=(0, 255, 0),
                                            matchesMask=mask_for_ratio_test,
                                            flags=0)
    plt.subplot(3, 1, 2)
    plt.imshow(passed_ratio_t_img)
    plt.title('The Matches that passed the ratio test')

    """draw the matches that failed the ratio test from the random 20 matches"""
    failed_ratio_t_img = cv2.drawMatchesKnn(img_left, kp_left, img_right, kp_right, matches_knn2,
                                            outImg=None,
                                            matchColor=(255, 0, 0),
                                            singlePointColor=(0, 255, 0),
                                            matchesMask=mask_discarded,
                                            flags=0)
    plt.subplot(3, 1, 3)
    plt.imshow(failed_ratio_t_img)
    plt.title('The Matches that discarded by the ratio test')

    plt.show()


if __name__ == '__main__':
    q1(0)
