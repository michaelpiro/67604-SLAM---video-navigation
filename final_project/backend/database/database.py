import os
from tqdm import tqdm
from final_project import Inputs
from final_project.algorithms.matching import (extract_kps_descs_matches, extract_inliers_outliers)
from final_project.algorithms.ransac import ransac_pnp_for_tracking_db
from final_project.algorithms.matching import MATCHER
from final_project.arguments import LEN_DATA_SET
import numpy as np
from final_project.backend.database.tracking_database import TrackingDB


def first_operation(db, frame_idx):
    """
    initiates the first operation for the database
    :param db: the db
    :param frame_idx: the frame index
    """
    l_prev_img, r_prev_img = Inputs.read_images(frame_idx)
    kp_l_prev, kp_r_prev, desc_l_prev, desc_r_prev, matches_prev = extract_kps_descs_matches(l_prev_img, r_prev_img)

    in_prev_idx = extract_inliers_outliers(kp_l_prev, kp_r_prev, matches_prev)[0]
    filtered_matches_prev = [matches_prev[i] for i in in_prev_idx]
    in_prev = np.array([True] * len(filtered_matches_prev))

    feature_prev, links_prev = db.create_links(desc_l_prev, kp_l_prev, kp_r_prev, filtered_matches_prev, in_prev)
    db.frameID_to_inliers_percent[frame_idx] = 100 * (len(in_prev_idx) / len(matches_prev))
    return feature_prev, links_prev


def create_db(start_frame=0, num_frames=200, db=None):
    """
    create a db
    :param path_to_sequence: the path to the sequence
    :param start_frame: the start frame usually 0
    :param num_frames: the number of frames
    :param save_every: the on each 500 images saves the database so that we will have a backup
    :param save_name: the name of the database
    :db: the db
    """
    if db is None:
        db = TrackingDB()
    if start_frame == 0:
        start_loop = 1
        feature_prev, links_prev = first_operation(db, 0)
        db.add_frame(links=links_prev, left_features=feature_prev, matches_to_previous_left=None, inliers=None)
    else:
        start_loop = start_frame
    for i in tqdm(range(start_loop, num_frames)):
        feature_cur, links_cur = first_operation(db, i)

        prev_features = db.features(i - 1)

        # extract matches of first left frame and the second left frame
        matches_l_l = MATCHER.match(prev_features, feature_cur)
        matches_l_l_backward = MATCHER.match(feature_cur, prev_features)
        if type(matches_l_l[0]) is tuple:
            matches_l_l = np.array(matches_l_l)
            matches_l_l = (matches_l_l[:, 0]).reshape(-1)
        else:
            matches_l_l = np.array(matches_l_l)
        if type(matches_l_l_backward[0]) is tuple:
            matches_l_l_backward = np.array(matches_l_l_backward)
            matches_l_l_backward = (matches_l_l_backward[:, 0]).reshape(-1)
        else:
            matches_l_l_backward = np.array(matches_l_l_backward)

        good_idx = []
        for j, match in enumerate(matches_l_l):
            feature_in_prev = match.queryIdx
            matched_feature_idx = match.trainIdx
            if matches_l_l_backward[matched_feature_idx].trainIdx != feature_in_prev:
                continue
            else:
                good_idx.append(j)

        good_idx = np.array(good_idx)
        filtered_matches_l_l = matches_l_l[good_idx]

        prev_links = db.all_frame_links(i - 1)
        best_matches_idx = ransac_pnp_for_tracking_db(filtered_matches_l_l, prev_links, links_cur,
                                                      db.frameID_to_inliers_percent[i])
        best_matches_idx = good_idx[best_matches_idx]

        in_prev_cur = np.array([False] * len(matches_l_l))
        in_prev_cur[best_matches_idx] = True

        db.add_frame(links_cur, feature_cur, matches_l_l, in_prev_cur)

    return db


def run(serialized_path):
    db = create_db(
        num_frames=LEN_DATA_SET,
        db=None,
    )
    db.serialize(serialized_path)
    return db


# data_base = TrackingDB()
# data_base.load("/Users/mac/67604-SLAM-video-navigation/final_project/SIFT_DB")
#
# bad_frames = [739, 1012, 1011, 1011, 1012, 1012, 1012, 1011, 1012]
# bad_tracks = [103896,
#               141814,
#               141929,
#               141930,
#               141931,
#               141932,
#               141933,
#               142057,
#               142104]
#
# for i in zip(bad_frames, bad_tracks):
#     link = data_base.link(i[0], i[1])
#     print(f"Frame: {i[0]}, Track: {i[1]}, link: {link}")

# run("/Users/mac/67604-SLAM-video-navigation/final_project/sift_db_after_change2")

# create_db(
#     num_frames=LEN_DATA_SET,
#     db=None)
