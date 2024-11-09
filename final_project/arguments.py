import os

DATA_PATH_WINDOWS = r'C:\Users\elyas\University\SLAM video navigation\VAN_ex\code\VAN_ex\dataset\sequences\00'
DATA_PATH_MAC = '/Users/mac/67604-SLAM-video-navigation/VAN_ex/dataset/sequences/00/'
DATA_HEAD_MAC = '/Users/mac/67604-SLAM-video-navigation/VAN_ex'
# DATA_PATH = r'C:\Users\elyas\University\SLAM video navigation\VAN_ex\code\VAN_ex\dataset\sequences\00\\'
DATA_HEAD_LINUX = r'/mnt/c/Users/elyas/University/AI/VAN_ex'
all_frames_serialized_db_path = DATA_HEAD_LINUX + "/docs/DB3000"
DATA_HEAD_WINDOWS = r'C:\Users\elyas\University\SLAM video navigation\VAN_ex\code\VAN_ex'
DATA_PATH_LINUX = DATA_HEAD_LINUX + '/dataset/sequences/00/'

BUNDLES_PATH = DATA_HEAD_MAC + '/docs/bundles_AKAZE'
LEN_DATA = len(os.listdir(DATA_PATH_MAC))
data_path = DATA_PATH_MAC
DATA_PATH = data_path
DATA_HEAD = DATA_HEAD_MAC

LEN_DATA_SET = len(os.listdir(DATA_PATH + 'image_0'))
AKAZE_PATH = DATA_HEAD + "/docs/SIFT"
SIFT_PATH = DATA_HEAD + "/docs/SIFT"
GROUND_TRUTH_PATH = DATA_HEAD + "/dataset/poses/00.txt"
SIFT_DB_PATH = "/Users/mac/67604-SLAM-video-navigation/final_project/SIFT_DB.pkl"
