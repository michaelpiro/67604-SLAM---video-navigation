import os

MICHAEL = '/Users/mac/67604-SLAM-video-navigation'
ELYASHIV = r'/mnt/c/Users/elyas/University/AI'
MAC = True
if MAC:
    START = MICHAEL
else:
    START = ELYASHIV

HEAD = START + "/VAN_ex"
DATA_PATH = HEAD + "/dataset/sequences/00/"
LEN_DATA_SET = len(os.listdir(DATA_PATH + '/image_0'))
GROUND_TRUTH_PATH = HEAD + "/dataset/poses/00.txt"

# GROUND_TRUTH_PATH = "/Users/mac/67604-SLAM-video-navigation/VAN_ex/dataset/poses/00.txt"

SIFT_DB_PATH = START + "/final_project/SIFT_DB" #mark bad
AKAZE_DB_PATH = START + "/final_project/AKAZE_DB" #mark bad
BUNDLES_PATH = HEAD + '/docs/bundles_AKAZE' #mark bad

TRANSFORMATIONS_NPY = "/Users/mac/67604-SLAM-video-navigation/final_project/pnp_global_transformations.npy"
PNP_GLOBAL_T_PATH = "/Users/mac/67604-SLAM-video-navigation/final_project/pnp_global_transformations.npy"

GRAPHS_DIR_PATH = START + "/final_project/graphs"