import cv2
import random
import matplotlib.pyplot as plt
import os


NUM_FEATURES_TO_SHOW = 20
MAX_FEATURES = 501
BAD_RATIO = 1
GOOD_RATIO = 0.4
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

    # draw keypoints on images
    kp_left_image = cv2.drawKeypoints(img_left, kp_left, 0, (0, 255, 0),
                                 flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,)

    kp_right_image = cv2.drawKeypoints(img_right, kp_right, 0, (0, 255, 0),
                                 flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    # show images

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(kp_left_image)
    plt.title('Left Image')

    plt.subplot(2, 1, 2)
    plt.imshow(kp_right_image)
    plt.title('Right Image')
    plt.show()


    # match keypoints
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_knn2 = matcher.knnMatch(desc_left, desc_right, k=2)

    matches_mask_1 = [[0,0] for i in range(len(matches_knn2))]
    matches_mask_2 = [[0, 0] for i in range(len(matches_knn2))]
    failed_indices = [0 for i in range(len(matches_knn2))]

    # find good matches
    for i, (m, n) in enumerate(matches_knn2):
        if m.distance < GOOD_RATIO * n.distance:
            failed_indices[i] = 1

    # print(sum(failed_indices))
    # select random 20 matches
    random_matches_1 = random.sample(range(len(matches_mask_1)), NUM_FEATURES_TO_SHOW)
    for i in random_matches_1:
        matches_mask_1[i] = [1, 0]
        if failed_indices[i] == 0:
            matches_mask_2[i] = [1, 0]


    total_matches = len(matches_knn2)
    good_matches_count = total_matches - sum(failed_indices)
    discarded_matches_count = sum(failed_indices)

    print(f"Ratio value used: {GOOD_RATIO}")
    print(f"Total matches: {total_matches}")
    print(f"Good matches: {good_matches_count}")
    print(f"Discarded matches: {discarded_matches_count}")


    #draw naive matches
    matches_img_knn1 = cv2.drawMatchesKnn(
                                        img_left, kp_left, img_right, kp_right, matches_knn2,
                                        outImg=None,
                                        matchColor=(0, 255, 0),
                                        singlePointColor=(255, 0, 0),
                                        matchesMask=matches_mask_1,
                                        flags=0)
    plt.imshow(matches_img_knn1)
    plt.title('Matches of knn1')
    plt.show()

    matches_img_knn2 = cv2.drawMatchesKnn(
                                        img_left, kp_left, img_right, kp_right, matches_knn2,
                                        outImg=None,
                                        matchColor=(0, 255, 0),
                                        singlePointColor=(255, 0, 0),
                                        matchesMask=matches_mask_1,
                                        flags=0)
    plt.imshow(matches_img_knn2)
    plt.title('Matches of knn2')
    plt.show()

    # #draw matches_of_knn1
    # matches_img_knn1 = cv2.drawMatches(img_left, kp_left, img_right, kp_right, matches_knn1, outImg=None, matchColor=(0, 155, 0),
    #                                     singlePointColor=(0, 255, 255), matchesMask=random_matches, flags=0)
    # cv2.imshow('Matches', matches_img_knn1)
    #
    #
    #
    #         #draw matches_of_knn2
    # # matches_img_knn2 = cv2.drawMatches(img_left, kp_left, img_right, kp_right, matches_knn2, outImg=None, matchColor=(0, 155, 0),
    # #                                     singlePointColor=(0, 255, 255), matchesMask=random_matches, flags=0)
    # cv2.imshow('Matches', matches_img_knn2)







# def detect_extract_keyPoints(idx):
#     img_left, img_right = read_images(idx)
#     # kp_left = feature.detect(img1)
#     # kp_right = feature.detect(img2)
#     kp_left,desc_left = feature.detectAndCompute(img_left, None)
#     kp_right,desc_right = feature.detectAndCompute(img_right, None)
#     kp_left_image = cv2.drawKeypoints(img_left, kp_left, None, color=(0, 255, 0), flags=0)
#     kp_right_image = cv2.drawKeypoints(img_right, kp_right, None, color=(0, 255, 0), flags=0)
#
#     # cv2.imshow('ORB', kp_image)
#     return kp_left, kp_right
#
# def extract_descriptors(kp_left, kp_right, idx):
#     img_left, img_right = read_images(idx)
#     kp_left, des_left = feature.compute(img_left, kp_left)
#     kp_right, des_right = feature.compute(img_right, kp_right)
#
#
#
#     return kp, des
if __name__ == '__main__':
    q1(0)

