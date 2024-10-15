import cv2

FEATURE = cv2.AKAZE_create(threshold=0.0008, nOctaves=4, nOctaveLayers=4)
bf_matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False)
MATCHER_LEFT_RIGHT = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
MATCHER = bf_matcher

