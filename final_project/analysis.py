import pickle
import random
from typing import Dict, List, Tuple

import gtsam
from matplotlib import pyplot as plt

from final_project import arguments
from final_project.algorithms.triangulation import linear_least_squares_triangulation
from final_project.backend.database.tracking_database import TrackingDB
from final_project.backend.GTSam.gtsam_utils import get_factor_symbols
from final_project.backend.GTSam.bundle import K_OBJECT, get_keyframes
from final_project.backend.GTSam.gtsam_utils import load

import numpy as np

SUBSET_FACTOR = 0.005

####################################################################################################
# TRACKING ANALYSIS
####################################################################################################
from final_project.utils import K, M2


def track_length(tracking_db: TrackingDB, trackId) -> int:
    return len(tracking_db.frames(trackId))


def total_number_of_tracks(tracking_db: TrackingDB) -> int:
    return tracking_db.track_num()


def number_of_frames(tracking_db: TrackingDB) -> int:
    return tracking_db.frame_num()


def mean_track_length(tracking_db: TrackingDB) -> float:
    track_ids = tracking_db.all_tracks()
    lengths = [track_length(tracking_db, trackId) for trackId in track_ids if
               track_length(tracking_db, trackId) > 1]
    return np.mean(lengths) if lengths else 0


def max_track_length(tracking_db: TrackingDB) -> int:
    track_ids = tracking_db.all_tracks()
    lengths = [track_length(tracking_db, trackId) for trackId in track_ids if
               track_length(tracking_db, trackId) > 1]
    return max(lengths) if lengths else 0


def min_track_length(tracking_db: TrackingDB) -> int:
    track_ids = tracking_db.all_tracks()
    lengths = [track_length(tracking_db, trackId) for trackId in track_ids if
               track_length(tracking_db, trackId) > 1]
    return min(lengths) if lengths else 0


def mean_number_of_frame_links(tracking_db: TrackingDB) -> float:
    if not tracking_db.frameId_to_trackIds_list:
        return 0
    return tracking_db.link_num() / tracking_db.frame_num()


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


def compute_matches_count(tracking_db: TrackingDB) -> Dict[int, int]:
    matches_count = {}
    for frameId in tracking_db.all_frames():
        matches_count[frameId] = len(tracking_db.all_frame_links(frameId))
    return matches_count


def plot_matches_count_graph(matches_count: Dict[int, int]):
    frames = sorted(matches_count.keys())
    counts = [matches_count[frame] for frame in frames]

    plt.figure(figsize=(10, 6))
    plt.plot(frames, counts)
    plt.xlabel('Frame ID')
    plt.ylabel('Number of Matches')
    plt.title('Matches Count Graph: Number of Matches per Frame')
    plt.grid(True)


def plot_inliers_percentage_graph(inliers_percentage_dict: Dict[int, float]):
    frames = sorted(inliers_percentage_dict.keys())
    percentages = [inliers_percentage_dict[frame] for frame in frames]

    plt.figure(figsize=(20, 10))
    plt.plot(frames, percentages)
    plt.xlabel('Frame ID')
    plt.ylabel('Percentage of Inliers')
    plt.title('Percentage of Inliers per Frame')
    plt.grid(True)


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


def plot_mean_factor_error(all_bundles):
    init_errors = []
    final_errors = []
    first_keyframes = []
    for bundle in all_bundles:
        result = bundle['result']
        graph = bundle['graph']
        initial_estimates = bundle['initial']
        keyframes = bundle['keyframes']

        first_keyframe = keyframes[0]
        num_factors = graph.size()

        init_errors.append(graph.error(initial_estimates) / num_factors)
        final_errors.append(graph.error(result) / num_factors)
        first_keyframes.append(first_keyframe)

    plt.figure(figsize=(10, 6))
    plt.plot(first_keyframes, init_errors, label='Initial Error')
    plt.plot(first_keyframes, final_errors, label='Final Error')
    plt.xlabel('First Keyframe ID')
    plt.ylabel('Mean Factor Error')
    plt.title('Mean Factor Error vs. First Keyframe ID')
    plt.legend()
    plt.grid(True)


def calculate_frame_projection_errors(db: TrackingDB, graph: gtsam.NonlinearFactorGraph, initial: gtsam.Values,
                                      result: gtsam.Values,
                                      first_keyframe):
    init_frame_errors = []
    final_frame_errors = []
    num_factors = graph.size()
    kf_cam_symbol = gtsam.symbol('c', first_keyframe)
    init_stereo_cam = gtsam.StereoCamera(initial.atPose3(kf_cam_symbol), K_OBJECT)
    final_stereo_cam = gtsam.StereoCamera(result.atPose3(kf_cam_symbol), K_OBJECT)

    # tracks = db.tracks(first_keyframe)
    for i in range(num_factors):
        factor = graph.at(i)
        if isinstance(factor, gtsam.GenericStereoFactor3D):
            camera_symbol, location_symbol = get_factor_symbols(factor)
            camera_number = int(gtsam.DefaultKeyFormatter(camera_symbol)[1:])
            if camera_number == first_keyframe:
                measurement = factor.measured()
                location_init = initial.atPoint3(location_symbol)
                location_final = result.atPoint3(location_symbol)

                projected_point_to_init = init_stereo_cam.project(location_init)
                projected_left_cam_init = projected_point_to_init.uL(), projected_point_to_init.v()
                # projected_right_cam = projected_point.uR(), projected_point.v()

                projected_point_to_final = final_stereo_cam.project(location_final)
                projected_left_cam_final = projected_point_to_final.uL(), projected_point_to_final.v()
                # projected_right_cam = projected_point.uR(), projected_point.v()

                measurement_left = measurement.uL(), measurement.v()
                # measurement_right = measurement.uR(), measurement.v()

                distance_left_init = round(
                    np.linalg.norm(np.array(measurement_left) - np.array(projected_left_cam_init)), 2)
                distance_left_final = round(
                    np.linalg.norm(np.array(measurement_left) - np.array(projected_left_cam_final)), 2)
                # distance_left = round(np.linalg.norm(np.array(measurement_left) - np.array(projected_left_cam)), 2)
                # distance_right = round(np.linalg.norm(np.array(measurement_right) - np.array(projected_right_cam)), 2)

                init_frame_errors.append(distance_left_init)
                final_frame_errors.append(distance_left_final)
    return init_frame_errors, final_frame_errors


def plot_median_projection_error(db, all_bundles):
    median_init_errors = []
    median_final_errors = []
    first_keyframes = []
    for bundle in all_bundles:
        result = bundle['result']
        graph = bundle['graph']
        initial_estimates = bundle['initial']
        keyframes = bundle['keyframes']

        first_keyframe = keyframes[0]
        init_frame_errors, final_frame_errors = calculate_frame_projection_errors(db, graph, initial_estimates, result,
                                                                                  first_keyframe)
        median_init_errors.append(np.median(init_frame_errors))
        median_final_errors.append(np.median(final_frame_errors))
        first_keyframes.append(first_keyframe)

    plt.figure(figsize=(10, 6))
    plt.plot(first_keyframes, median_init_errors, label='Initial Error')
    plt.plot(first_keyframes, median_final_errors, label='Final Error')
    plt.xlabel('First Keyframe ID')
    plt.ylabel('Median Projection Error')
    plt.title('Median Projection Error vs. First Keyframe ID')
    plt.legend()
    plt.grid(True)


def map_camera_to_bundle(camera: int, keyframes):
    for i, keyframe in enumerate(keyframes):

        first_keyframe = keyframe[0]
        second_keyframe = keyframe[1]
        if first_keyframe <= camera <= second_keyframe:
            return i
    return None


def get_projection_errors(tracks, bundles, keyframes, db: TrackingDB):
    distance_dict = {}
    reprojection_erros_left = {}
    reprojection_erros_right = {}

    for trackId in tracks:
        frames = db.frames(trackId)
        track_first_frame = min(frames)
        for frame in frames:
            bundle_idx = map_camera_to_bundle(frame, keyframes)
            bundle = bundles[bundle_idx]
            result = bundle['result']
            point_symbol = gtsam.symbol('l', trackId)
            camera_symbol = gtsam.symbol('c', frame)
            try:
                point = result.atPoint3(point_symbol)
                camera = result.atPose3(camera_symbol)
            except Exception as e:
                continue
            # point = result.atPoint3(point_symbol)
            # camera = result.atPose3(camera_symbol)

            stereo_cam = gtsam.StereoCamera(camera, K_OBJECT)
            projected_point = stereo_cam.project(point)
            projected_left_cam = projected_point.uL(), projected_point.v()
            projected_right_cam = projected_point.uR(), projected_point.v()
            link = db.link(frame, trackId)

            distance_left = round(np.linalg.norm(np.array((link.x_left, link.y)) - np.array(projected_left_cam)), 2)
            distance_right = round(np.linalg.norm(np.array((link.x_right, link.y)) - np.array(projected_right_cam)), 2)

            frame_dist = int(frame - track_first_frame)
            if frame_dist not in reprojection_erros_left:
                reprojection_erros_left[frame_dist] = []
                reprojection_erros_right[frame_dist] = []
            reprojection_erros_left[frame_dist].append(distance_left)
            reprojection_erros_right[frame_dist].append(distance_right)

    dist_keys = list(reprojection_erros_left.keys())
    for key in dist_keys:
        distance_dict[key] = np.median(reprojection_erros_left[key]), np.median(reprojection_erros_right[key])
    return reprojection_erros_left, reprojection_erros_right


def plot_reprojection_error_vs_track_length(tracking_db: TrackingDB, bundles, keyframes):
    def get_tracks_subset(db: TrackingDB, subset_size: int):
        all_tracks = db.all_tracks()
        eligible_tracks = [track_Id for track_Id in all_tracks if track_length(db, track_Id) > 1]
        if not eligible_tracks:
            return None
        return np.random.choice(eligible_tracks, subset_size)

    def read_kth_camera(k):
        filename = '/Users/mac/67604-SLAM-video-navigation/VAN_ex/dataset/poses/00.txt'
        with open(filename, 'r') as file:
            for current_line_number, line in enumerate(file, start=0):
                if current_line_number == k:
                    camera = line.strip()
                    break
        numbers = list(map(float, camera.split()))
        matrix = np.array(numbers).reshape(3, 4)
        return matrix

    def plot_reprojection_errors(reprojection_errors: Dict[int, Tuple[float, float]]):
        the_frames = sorted(reprojection_errors.keys())
        left_errors = [reprojection_errors[frame][0] for frame in the_frames]
        right_errors = [reprojection_errors[frame][1] for frame in the_frames]

        plt.figure(figsize=(10, 6))
        plt.plot(the_frames, left_errors, label='Left Camera')
        plt.plot(the_frames, right_errors, label='Right Camera')
        plt.xlabel('distance from reference frame')
        plt.ylabel('projection Error')
        plt.xlim(0, max(the_frames))
        plt.ylim(0, 8)
        plt.title('projection Error vs track length')
        plt.legend()
        plt.grid(True)

    num_tracks = int(tracking_db.track_num() * SUBSET_FACTOR)
    print(f'num_tracks: {num_tracks}')
    track_ids = get_tracks_subset(tracking_db, num_tracks)
    distance_dict = {}
    reprojection_erros_left = {}
    reprojection_erros_right = {}
    for trackId in track_ids:
        track_last_frame = tracking_db.last_frame_of_track(trackId)
        frames = tracking_db.frames(trackId)
        left_camera_mat = read_kth_camera(track_last_frame)
        link = tracking_db.link(track_last_frame, trackId)

        p = K @ left_camera_mat
        q = K @ M2 @ np.vstack((left_camera_mat, np.array([0, 0, 0, 1])))

        world_point = linear_least_squares_triangulation(p, q, (link.x_left, link.y), (link.x_right, link.y))
        world_point_4d = np.append(world_point, 1).reshape(4, 1)

        projections = {}

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
            frame_dist = int(track_last_frame - frameId)
            if frame_dist not in reprojection_erros_left:
                reprojection_erros_left[frame_dist] = []
                reprojection_erros_right[frame_dist] = []
            reprojection_erros_left[int(track_last_frame - frameId)].append(
                np.linalg.norm(points_vec_left - projection_left[0:2]))
            reprojection_erros_right[int(track_last_frame - frameId)].append(
                np.linalg.norm(points_vec_right - projection_right[0:2]))

    dist_keys = list(reprojection_erros_left.keys())
    for key in dist_keys:
        distance_dict[key] = np.median(reprojection_erros_left[key]), np.median(reprojection_erros_right[key])

    plot_reprojection_errors(distance_dict)

    rep_left, rep_right = get_projection_errors(track_ids, bundles, keyframes, db)
    dist_keys = list(rep_left.keys())
    for key in dist_keys:
        distance_dict[key] = np.median(rep_left[key]), np.median(rep_right[key])

    plot_reprojection_errors(distance_dict)


def run_analysis(db, bundles_list):
    total_tracks = total_number_of_tracks(db)
    num_frames = number_of_frames(db)
    mean_length = mean_track_length(db)
    max_length = max_track_length(db)
    min_length = min_track_length(db)
    mean_frame_links = mean_number_of_frame_links(db)

    print(f"Total number of tracks: {total_tracks}")
    print(f"Total number of frames: {num_frames}")
    print(f"Mean track length: {mean_length}, Max track length: {max_length}, Min track length: {min_length}")
    print(f"Mean number of frame links: {mean_frame_links}")

    # Compute outgoing tracks
    outgoing_tracks = compute_outgoing_tracks(db)

    # Plot the connectivity graph
    plot_connectivity_graph(outgoing_tracks)

    # Compute matches count
    matches_count = compute_matches_count(db)

    # Plot the matches count graph
    plot_matches_count_graph(matches_count)

    # Plot the inliers percentage graph
    plot_inliers_percentage_graph(db.frameID_to_inliers_percent)

    # Calculate track lengths
    track_lengths = calculate_track_lengths(db)

    # Plot the track length histogram
    plot_track_length_histogram(track_lengths)

    # Plot the reprojection error vs track length
    plot_reprojection_error_vs_track_length(db)

    # bundles_dict, graphs, results = load(bundles_path)
    # bundles = []
    # bundles_keys = sorted(list(bundles_dict.keys()))
    #
    # for key in bundles_keys:
    #     bundles.append(bundles_dict[key])

    # Plot mean factor error
    plot_mean_factor_error(bundles_list)

    # Plot median projection error
    plot_median_projection_error(db, bundles_list)

    plt.show()

def load_bundles_list(base_filename):
    """ load TrackingDB to base_filename+'.pkl' file. """
    filename = base_filename + '.pkl'

    with open(filename, 'rb') as file:
        data = pickle.load(file)
        bundles= data
    print('Bundles loaded from', filename)
    return bundles

if __name__ == '__main__':
    db = TrackingDB()
    # serialized_path = arguments.DATA_HEAD + "/docs/AKAZE/db/db_3359"
    # serialized_path = "/Users/mac/67604-SLAM-video-navigation/VAN_ex/docs/ORB/db/db_500"
    serialized_path = "/Users/mac/67604-SLAM-video-navigation/VAN_ex/docs/SIFT/db/db_3359"
    # bundles_path = "/Users/mac/67604-SLAM-video-navigation/VAN_ex/docs/bundles_AKAZE"
    db.load(serialized_path)
    plt.show()
    key_frames = get_keyframes(db)
    bundles = load_bundles_list("/Users/mac/67604-SLAM-video-navigation/final_project/SIFT_BUNDLES")
    plot_reprojection_error_vs_track_length(db, bundles, key_frames)
    plt.show()
    # run_analysis(db, bundles)
