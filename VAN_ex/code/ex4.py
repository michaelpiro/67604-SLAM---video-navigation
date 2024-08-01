import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
from typing import List, Tuple, Dict, Sequence, Optional
from timeit import default_timer as timer
from tracking_database import TrackingDB, Link, MatchLocation
from tqdm import tqdm
# from ex4_2_to_6 import q_4_2, q_4_3, q_4_4, q_4_5, q_4_6, q_4_7, find_longest_track_frames

import ex3
import ex2
import random

NO_ID = -1

FEATURE = cv2.AKAZE_create()
FEATURE.setNOctaves(4)
# FEATURE.setThreshold(0.01)
print(FEATURE.getThreshold())
MATCHER = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False)
# DATA_PATH = r'C:\Users\elyas\University\SLAM video navigation\VAN_ex\code\VAN_ex\dataset\sequences\00\\'
DATA_PATH = '/Users/mac/67604-SLAM-video-navigation/VAN_ex/dataset/sequences/00/'
LEN_DATA_SET = len(os.listdir(DATA_PATH + 'image_0'))
# LEN_DATA_SET = 1000
P = ex3.P
Q = ex3.Q
K = ex3.K
M1 = ex3.M1
M2 = ex3.M2


def extract_kps_descs_matches(img_0, img1):
    kp0, desc0 = FEATURE.detectAndCompute(img_0, None)
    kp1, desc1 = FEATURE.detectAndCompute(img1, None)
    matches = MATCHER.match(desc0, desc1)
    return kp0, kp1, desc0, desc1, matches


def triangulate_last_frame(tracking_db: TrackingDB, p, q, links=None):
    """
    Triangulate the matched points using OpenCV
    :param inliers:
    :return:
    """
    if links is None:
        links = tracking_db.all_last_frame_links()
    x = np.zeros((len(links), 3))
    for i in range(len(links)):
        x_left, x_right, y = links[i].x_left, links[i].x_right, links[i].y
        p_left, p_right = (x_left, y), (x_right, y)
        x[i] = ex2.linear_least_squares_triangulation(p, q, p_left, p_right)
    return x


def find_concensus_points_and_idx(matches_lr_0, in_lr_0, matches_l0_l1, matches_lr1, in_lr1):
    if len(matches_lr_0) != len(in_lr_0):
        print("bad")
    assert len(matches_lr_0) == len(in_lr_0)
    assert len(matches_lr1) == len(in_lr1)
    # Create dictionaries for quick lookup
    # a dict that it's keys are the keypoint of the match in the left image,
    # and the value is the index of the match in the matches array
    dict_lr_0 = {match.queryIdx: i for i, match in enumerate(matches_lr_0) if in_lr_0[i]}
    dict_lr_1 = {match.queryIdx: i for i, match in enumerate(matches_lr1) if in_lr1[i]}

    con = []
    matches = []
    matches_l0_l1_good_idx = []
    for i, match_l_l in enumerate(matches_l0_l1):
        kp_ll_0 = match_l_l.queryIdx
        kp_ll_1 = match_l_l.trainIdx

        # Check if the match exists in both dictionaries
        if kp_ll_0 in dict_lr_0 and kp_ll_1 in dict_lr_1:
            i0 = dict_lr_0[kp_ll_0]
            i1 = dict_lr_1[kp_ll_1]
            con.append((i0, i1))
            matches.append((matches_lr_0[i0], matches_lr1[i1]))
            matches_l0_l1_good_idx.append(i)
        else:
            pass
            # print("bad match")
    # print(f"good ratio: {len(matches) / len(matches_l0_l1)}")

    return np.array(con), np.array(matches), matches_l0_l1_good_idx


def transformation_agreement(T, traingulated_pts, prev_left_pix_values, prev_right_pix_values,
                             ordered_cur_left_pix_values, ordered_cur_right_pix_values, x_condition=True):
    T_4x4 = np.vstack((T, np.array([0, 0, 0, 1])))
    points_4d = np.hstack((traingulated_pts, np.ones((traingulated_pts.shape[0], 1)))).T
    l1_T = K @ T
    l1_4d_points = (T_4x4 @ points_4d)
    to_the_right = (K @ M2)

    transform_to_l0_points = (P @ points_4d).T
    transform_to_l0_points = transform_to_l0_points / transform_to_l0_points[:, 2][:, np.newaxis]
    real_y = prev_left_pix_values[:, 1]
    y_diff = transform_to_l0_points[:, 1] - real_y
    agree_l0 = np.abs(transform_to_l0_points[:, 1] - real_y) < 2

    transform_to_r0_points = (to_the_right @ points_4d).T
    transform_to_r0_points = transform_to_r0_points / transform_to_r0_points[:, 2][:, np.newaxis]
    real_y = prev_right_pix_values[:, 1]
    agree_r0 = np.abs(transform_to_r0_points[:, 1] - real_y) < 2
    if x_condition:
        real_x_l = prev_left_pix_values[:, 0]
        real_x_r = prev_right_pix_values[:, 0]
        cond_x = real_x_l > real_x_r + 2
    else:
        cond_x = np.ones_like(agree_r0)
    agree_0 = np.logical_and(agree_r0, cond_x, agree_l0)

    transformed_to_l1_points = (K @ l1_4d_points[:3, :]).T
    transformed_to_l1_points = transformed_to_l1_points / transformed_to_l1_points[:, 2][:, np.newaxis]
    real_y = ordered_cur_left_pix_values[:, 1]
    agree_l1 = np.abs(transformed_to_l1_points[:, 1] - real_y) < 2

    transformed_to_r1_points = (to_the_right @ l1_4d_points).T
    transformed_to_r1_points = transformed_to_r1_points / transformed_to_r1_points[:, 2][:, np.newaxis]
    real_y = ordered_cur_right_pix_values[:, 1]
    agree_r1 = np.abs(transformed_to_r1_points[:, 1] - real_y) < 2
    if x_condition:
        real_x_l = ordered_cur_left_pix_values[:, 0]
        real_x_r = ordered_cur_right_pix_values[:, 0]
        cond_x = real_x_l > real_x_r + 2
    else:
        cond_x = np.ones_like(agree_r1)
    agree_1 = np.logical_and(agree_r1, cond_x, agree_l1)
    return np.logical_and(agree_0, agree_1)


def ransac_pnp_for_tracking_db(traingulated_pts, matches, links_prev, links_cur):
    """ Perform RANSAC to find the best transformation"""
    best_inliers = 0
    best_T = None
    best_matches_idx = None
    diff_coeff = np.zeros((5, 1))
    prev_left_pix_values = []
    prev_right_pix_values = []
    ordered_cur_left_pix_values = []
    ordered_cur_right_pix_values = []
    for link in links_prev:
        prev_left_pix_values.append((link.x_left, link.y))
        prev_right_pix_values.append((link.x_right, link.y))
    for match in matches:
        link_index = match.trainIdx
        ordered_cur_left_pix_values.append((links_cur[link_index].x_left, links_cur[link_index].y))
        ordered_cur_right_pix_values.append((links_cur[link_index].x_right, links_cur[link_index].y))
    prev_left_pix_values = np.array(prev_left_pix_values)
    prev_right_pix_values = np.array(prev_right_pix_values)
    ordered_cur_left_pix_values = np.array(ordered_cur_left_pix_values)
    ordered_cur_right_pix_values = np.array(ordered_cur_right_pix_values)

    for i in range(ex3.RANSAC_ITERATIONS):

        # Randomly select 4 points in the world coordinate system
        random_idx = np.random.choice(len(traingulated_pts), 4, replace=False)
        random_world_points = traingulated_pts[random_idx]
        random_cur_l_pixels = ordered_cur_left_pix_values[random_idx]

        # solve PnP problem to get the transformation
        success, rotation_vector, translation_vector = cv2.solvePnP(random_world_points, random_cur_l_pixels, K,
                                                                    distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)
        if success:
            T = ex3.rodriguez_to_mat(rotation_vector, translation_vector)
        else:
            continue

        points_agreed = transformation_agreement(T, traingulated_pts, prev_left_pix_values, prev_right_pix_values,
                                                 ordered_cur_left_pix_values, ordered_cur_right_pix_values,
                                                 x_condition=True)

        inliers_idx = np.where(points_agreed == True)
        if np.sum(points_agreed) > best_inliers:
            best_inliers = np.sum(points_agreed)
            best_T = T
            best_matches_idx = inliers_idx

    return best_T, best_matches_idx, ordered_cur_left_pix_values[best_matches_idx[0]]


def find_best_transformation_ex4(traingulated_pts, matches, links_prev, links_cur):
    """ Find the best transformation using RANSAC"""

    T, inliers_idx, agreed_img_points = ransac_pnp_for_tracking_db(traingulated_pts, matches, links_prev, links_cur)

    best_matches = matches[inliers_idx[0]]

    diff_coeff = np.zeros((5, 1))
    pt_3d = traingulated_pts[inliers_idx[0]]
    if len(pt_3d) < 4:
        raise ValueError("Not enough points to estimate the transformation")
    success, rotation_vector, translation_vector = cv2.solvePnP(pt_3d, agreed_img_points, K,
                                                                distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)
    if success:
        return ex3.rodriguez_to_mat(rotation_vector, translation_vector), inliers_idx[0]
    return None


def create_DB(path_to_sequence=r"VAN_ex/code/VAN_ex/dataset/sequences/00", num_of_frames=50):
    l_prev_img, r_prev_img = ex2.read_images(0)
    kp_l_prev, kp_r_prev, desc_l_prev, desc_r_prev, matches_prev = extract_kps_descs_matches(l_prev_img, r_prev_img)
    DB = TrackingDB()

    in_prev = np.array([False] * len(matches_prev))
    in_prev_idx = ex3.extract_inliers_outliers(kp_l_prev, kp_r_prev, matches_prev)[0]
    in_prev[in_prev_idx] = True
    feature_prev, links_prev = DB.create_links(desc_l_prev, kp_l_prev, kp_r_prev, matches_prev, in_prev)
    DB.add_frame(links=links_prev, left_features=feature_prev, matches_to_previous_left=None, inliers=None)
    DB.frameID_to_inliers_percent[0] = 100 * (len(in_prev_idx) / len(matches_prev))

    for i in tqdm(range(1, num_of_frames)):
        # load the next frames and extract the keypoints and descriptors
        img_l_cur, img_r_cur = ex2.read_images(i)
        kp_l_cur, kp_r_cur, desc_l_cur, desc_r_cur, matches_cur = extract_kps_descs_matches(img_l_cur, img_r_cur)

        # extract the inliers and outliers and triangulate the points
        in_cur_idx = ex3.extract_inliers_outliers(kp_l_cur, kp_r_cur, matches_cur)[0]
        DB.frameID_to_inliers_percent[i] = 100 * (len(in_cur_idx) / len(matches_cur))
        in_cur = np.array([False] * len(matches_cur))
        in_cur[in_cur_idx] = True

        # create the links for the curr frame:
        feature_cur, links_cur = DB.create_links(desc_l_cur, kp_l_cur, kp_r_cur, matches_cur, in_cur)

        # extract matches of first left frame and the second left frame
        matches_l_l = MATCHER.match(feature_prev, feature_cur)
        LEN_FEATURES_PREV = len(feature_prev)
        LEN_FEATURES_CUR = len(feature_cur)
        LEN_MATCHES = len(matches_l_l)
        inliers_cond = np.array([False] * LEN_MATCHES)
        # matches_l_l = np.array(MATCHER.knnMatch(desc_l_prev, desc_l_cur, k=2))
        if type(matches_l_l[0]) is tuple:
            matches_l_l = np.array(matches_l_l)
            matches_l_l = matches_l_l[:, 0]
        else:
            matches_l_l = np.array(matches_l_l)
        matches_cur = np.array(matches_cur)
        matches_prev = np.array(matches_prev)
        con, matches, matches_l0_l1_good_idx = find_concensus_points_and_idx(matches_prev, in_prev, matches_l_l,
                                                                             matches_cur, in_cur)
        matches_l0_l1_good_idx = np.array(matches_l0_l1_good_idx)
        # good_matches = matches_l_l[matches_l0_l1_good_idx]

        traingulated_pts = triangulate_last_frame(DB, P, Q)
        temp_traingulated_pts = traingulated_pts[matches_l0_l1_good_idx]
        temp_links_prev = np.array(links_prev)[matches_l0_l1_good_idx]
        temp_matches_l_l = matches_l_l[matches_l0_l1_good_idx]

        relative_transformation, idx = find_best_transformation_ex4(temp_traingulated_pts, temp_matches_l_l,
                                                                    temp_links_prev,
                                                                    links_cur)
        # relative_transformation, idx = find_best_transformation_ex4(traingulated_pts, matches_l_l, links_prev,
        #                                                             links_cur)

        # todo it needs to be so that the inliers would be a binary array for the matches of the l to l (matches l_l who are best
        # final_matches = (matches_pairs[:, 0])[idx]
        # final_matches = [match.queryIdx for match in final_matches]
        in_prev_cur = np.array([False] * len(matches_l_l))
        temp_idx = matches_l0_l1_good_idx[idx]
        in_prev_cur[temp_idx] = True
        # in_prev_cur[idx] = True
        DB.add_frame(links_cur, feature_cur, temp_matches_l_l, in_prev_cur)

        # update the keypoints, descriptors and matches
        kp_l_prev, kp_r_prev, desc_l_prev, desc_r_prev, matches_prev = kp_l_cur, kp_r_cur, desc_l_cur, \
                                                                       desc_r_prev, matches_cur
        feature_prev = feature_cur
        in_prev = in_cur
        # needs to be only matches from l to prev l who are good
        links_prev = links_cur
    return DB


def q_4_2(tracking_db: TrackingDB):
    def track_length(tracking_db: TrackingDB, trackId) -> int:
        return len(tracking_db.frames(trackId))

    def total_number_of_tracks(tracking_db: TrackingDB) -> int:
        return len(
            [trackId for trackId in tracking_db.trackId_to_frames if track_length(tracking_db, trackId) > 1])

    def number_of_frames(tracking_db: TrackingDB) -> int:
        return len(tracking_db.frameId_to_trackIds_list)

    def mean_track_length(tracking_db: TrackingDB) -> float:
        lengths = [track_length(tracking_db, trackId) for trackId in tracking_db.trackId_to_frames if
                   track_length(tracking_db, trackId) > 1]
        return np.mean(lengths) if lengths else 0

    def max_track_length(tracking_db: TrackingDB) -> int:
        lengths = [track_length(tracking_db, trackId) for trackId in tracking_db.trackId_to_frames if
                   track_length(tracking_db, trackId) > 1]
        return max(lengths) if lengths else 0

    def min_track_length(tracking_db: TrackingDB) -> int:
        lengths = [track_length(tracking_db, trackId) for trackId in tracking_db.trackId_to_frames if
                   track_length(tracking_db, trackId) > 1]
        return min(lengths) if lengths else 0

    def mean_number_of_frame_links(tracking_db: TrackingDB) -> float:
        if not tracking_db.frameId_to_trackIds_list:
            return 0
        total_links = sum(len(trackIds) for trackIds in tracking_db.frameId_to_trackIds_list.values())
        return total_links / len(tracking_db.frameId_to_trackIds_list)

    total_tracks = total_number_of_tracks(tracking_db)
    num_frames = number_of_frames(tracking_db)
    mean_length = mean_track_length(tracking_db)
    max_length = max_track_length(tracking_db)
    min_length = min_track_length(tracking_db)
    mean_frame_links = mean_number_of_frame_links(tracking_db)

    print(f"Total number of tracks: {total_tracks}")
    print(f"Total number of frames: {num_frames}")
    print(f"Mean track length: {mean_length}, Max track length: {max_length}, Min track length: {min_length}")
    print(f"Mean number of frame links: {mean_frame_links}")


def q_4_3(tracking_db: TrackingDB):
    def get_feature_location(tracking_db: TrackingDB, frameId: int, trackId: int) -> Tuple[float, float]:
        link = tracking_db.linkId_to_link[(frameId, trackId)]
        return link.x_left, link.y

    def find_random_track_of_length(tracking_db: TrackingDB, length: int) -> Optional[int]:
        eligible_tracks = [trackId for trackId, frames in tracking_db.trackId_to_frames.items() if
                           len(frames) >= length]
        if not eligible_tracks:
            return None
        return random.choice(eligible_tracks)

    def visualize_track(tracking_db: TrackingDB, trackId: int):
        frames = tracking_db.frames(trackId)
        print(f"Track {trackId} has {len(frames)} frames")
        plt.figure()
        for i in range(0, 6, 1):
            # print(f"Frame {frames[i]}")
            frameId = frames[i]
            img, _ = ex2.read_images(frameId)
            x_left, y = get_feature_location(tracking_db, frameId, trackId)
            x_min = int(max(x_left - 10, 0))
            x_max = int(min(x_left + 10, img.shape[1]))
            y_min = int(max(y - 10, 0))
            y_max = int(min(y + 10, img.shape[0]))
            cutout = img[y_min:y_max, x_min:x_max]

            plt.subplot(6, 2, 2 * i + 1)
            plt.imshow(img, cmap='gray')
            plt.scatter(x_left, y, color='red')  # Center of the cutout

            plt.subplot(6, 2, 2 * i + 2)
            plt.imshow(cutout, cmap='gray')
            plt.scatter([10], [10], color='red', marker='x', linewidths=1)  # Center of the cutout
            if i == 0:
                plt.title(f"Frame {frameId}, Track {trackId}")
        # plt.show()

    minimal_length = 6
    trackId = find_random_track_of_length(tracking_db, minimal_length)
    if trackId is None:
        print(f"No track of length {minimal_length} found")
    else:
        print(f"Track of length {minimal_length} found: {trackId}")
        visualize_track(tracking_db, trackId)


def q_4_4(tracking_db: TrackingDB):
    def compute_outgoing_tracks(tracking_db: TrackingDB) -> Dict[int, int]:
        outgoing_tracks = {}
        for frameId in sorted(tracking_db.frameId_to_trackIds_list.keys()):
            next_frameId = frameId + 1
            if next_frameId in tracking_db.frameId_to_trackIds_list:
                current_tracks = set(tracking_db.frameId_to_trackIds_list[frameId])
                next_tracks = set(tracking_db.frameId_to_trackIds_list[next_frameId])
                outgoing_tracks[frameId] = len(current_tracks.intersection(next_tracks))
            else:
                outgoing_tracks[frameId] = 0
        return outgoing_tracks

    def plot_connectivity_graph(outgoing_tracks: Dict[int, int]):
        frames = sorted(outgoing_tracks.keys())
        counts = [outgoing_tracks[frame] for frame in frames]

        plt.figure(figsize=(10, 6))
        plt.plot(frames, counts)
        plt.xlabel('Frame ID')
        plt.ylabel('Number of Outgoing Tracks')
        plt.title('Connectivity Graph: Outgoing Tracks per Frame')
        plt.grid(True)
        # plt.show()

    # Compute outgoing tracks
    outgoing_tracks = compute_outgoing_tracks(tracking_db)

    # Plot the connectivity graph
    plot_connectivity_graph(outgoing_tracks)


def q_4_5(tracking_db: TrackingDB):
    def plot_inliers_percentage_graph(inliers_percentage_dict: Dict[int, float]):
        frames = sorted(inliers_percentage_dict.keys())
        percentages = [inliers_percentage_dict[frame] for frame in frames]

        plt.figure(figsize=(20, 10))
        plt.plot(frames, percentages)
        plt.xlabel('Frame ID')
        plt.ylabel('Percentage of Inliers')
        plt.title('Percentage of Inliers per Frame')
        plt.grid(True)
        # plt.show()

    # inliers_percentage = {}
    # for frame_idx in range(LEN_DATA_SET):
    #     img_l, img_r = ex2.read_images(frame_idx)
    #     kp0, kp1, desc0, desc1, matches = ex3.extract_kps_descs_matches(img_l, img_r)
    #     inliers, outliers = ex3.extract_inliers_outliers(kp0, kp1, matches)
    #     inliers_percentage[frame_idx] = (len(inliers) / (len(inliers) + len(outliers))) * 100
    # Compute inliers percentage
    # inliers_percentage = compute_inliers_percentage(tracking_db)

    # Plot the inliers percentage graph
    plot_inliers_percentage_graph(tracking_db.frameID_to_inliers_percent)


def q_4_6(tracking_db: TrackingDB):
    def calculate_track_lengths(tracking_db: TrackingDB) -> List[int]:
        track_lengths = [len(frames) for trackId, frames in tracking_db.trackId_to_frames.items()]
        return track_lengths

    def plot_track_length_histogram(track_lengths: List[int]):
        plt.figure(figsize=(10, 6))
        plt.hist(track_lengths, bins=range(1, max(track_lengths) + 2), edgecolor='black')
        plt.xlabel('Track Length')
        plt.ylabel('Frequency')
        plt.title('Track Length Histogram')
        plt.grid(True)

        plt.yscale('log')
        plt.xlim((0, 150))
        # plt.show()

    # Calculate track lengths
    track_lengths = calculate_track_lengths(tracking_db)

    # Plot the track length histogram
    plot_track_length_histogram(track_lengths)


def q_4_7(tracking_db: TrackingDB):
    def find_random_track_of_length(tracking_db: TrackingDB, length: int) -> Optional[int]:
        eligible_tracks = [trackId for trackId, frames in tracking_db.trackId_to_frames.items() if
                           len(frames) >= length]
        if not eligible_tracks:
            return None
        return random.choice(eligible_tracks)

    def read_kth_camera(k):
        filename = '/Users/mac/67604-SLAM-video-navigation/VAN_ex/dataset/poses/00.txt'
        with open(filename, 'r') as file:
            for current_line_number, line in enumerate(file, start=1):
                if current_line_number == k:
                    camera = line.strip()
                    break
        numbers = list(map(float, camera.split()))
        matrix = np.array(numbers).reshape(3, 4)
        return matrix

    def plot_reprojection_errors(reprojection_errors: Dict[int, Tuple[float, float]]):
        frames = sorted(reprojection_errors.keys())
        left_errors = [reprojection_errors[frame][0] for frame in frames]
        right_errors = [reprojection_errors[frame][1] for frame in frames]

        plt.figure(figsize=(10, 6))
        plt.plot(frames, left_errors, label='Left Camera')
        plt.plot(frames, right_errors, label='Right Camera')
        plt.xlabel('distance from reference frame')
        plt.ylabel('projection Error')
        plt.title('projection Error vs track length')
        plt.legend()
        plt.grid(True)
        # plt.show()

    trackId = find_random_track_of_length(tracking_db, 10)
    track_last_frame = tracking_db.last_frame_of_track(trackId)
    frames = tracking_db.frames(trackId)
    left_camera_mat = read_kth_camera(track_last_frame)
    link = tracking_db.link(track_last_frame, trackId)

    p = K @ left_camera_mat
    q = K @ M2 @ np.vstack((left_camera_mat, np.array([0, 0, 0, 1])))

    world_point = ex2.linear_least_squares_triangulation(p, q, (link.x_left, link.y), (link.x_right, link.y))
    world_point_4d = np.append(world_point, 1).reshape(4, 1)

    projections = {}
    reprojection_erros = {}
    for frameId in frames:
        left_camera_mat = read_kth_camera(frameId)
        p = K @ left_camera_mat
        q = K @ M2 @ np.vstack((left_camera_mat, np.array([0, 0, 0, 1])))
        projection_left, projection_right = ((p @ world_point_4d).T, (q @ world_point_4d).T)

        projection_left = projection_left / projection_left[:, 2][:, np.newaxis]
        projection_right = projection_right / projection_right[:, 2][:, np.newaxis]
        projections[frameId] = (projection_left, projection_right)
        link = tracking_db.link(frameId, trackId)
        points_vec_left = np.array([link.x_left, link.y, 1])
        points_vec_right = np.array([link.x_right, link.y, 1])
        reprojection_erros[int(track_last_frame - frameId)] = (np.linalg.norm(points_vec_left - projection_left[0:2]),
                                                               np.linalg.norm(points_vec_right - projection_right[0:2]))

    plot_reprojection_errors(reprojection_erros)


if __name__ == '__main__':
    # print(M1)
    # print(M2)
    # print(K)
    # all_frames_serialized_db_path = "/Users/mac/67604-SLAM-video-navigation/VAN_ex/docs/DB3000"
    # serialized_db_path = "/Users/mac/67604-SLAM-video-navigation/VAN_ex/docs/DB_all_after_changing the percent"
    # path = r"C:\Users\elyas\University\SLAM video navigation\VAN_ex\code\VAN_ex\dataset\sequences\00"
    # serialized_path = "/Users/mac/67604-SLAM-video-navigation/VAN_ex/docs/DB_all_after_changing the percent"
    # serialized_path = "/Users/mac/67604-SLAM-video-navigation/VAN_ex/docs/check50"
    serialized_path = "/Users/mac/67604-SLAM-video-navigation/VAN_ex/docs/NEWEST"
    path = r"C:\Users\elyas\University\SLAM video navigation\VAN_ex\code\VAN_ex\dataset\sequences\00"
    db = create_DB(num_of_frames=500)
    db.serialize(serialized_path)

    db = TrackingDB()
    db.load(serialized_path)
    print(len(db.frames(7)))
    print(db.all_tracks())
    for track in db.all_tracks():
        if len(db.frames(track)) < 2:
            print(len(db.frames(track)))

    # q_4_1(db)
    # q_4_2(db)
    # q_4_3(db)
    # q_4_4(db)
    # q_4_5(db)
    # q_4_6(db)
    # q_4_7(db)
    # plt.show()
