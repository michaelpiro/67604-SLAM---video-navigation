import os
import gtsam
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List, Tuple, Dict, Sequence, Optional

import arguments
import random
from tracking_database import TrackingDB
from ex4 import K, M2, M1, P, Q, triangulate_last_frame
from gtsam.utils.plot import plot_trajectory, plot_3d_points, set_axes_equal
from ex3 import rodriguez_to_mat, read_extrinsic_matrices
from ex2 import read_images

BASE_LINE_SIGN = -1

# DATA_PATH = r'C:\Users\elyas\University\SLAM video navigation\VAN_ex\code\VAN_ex\dataset\sequences\00\\'
DATA_PATH = arguments.DATA_PATH
LEN_DATA_SET = len(os.listdir(DATA_PATH + 'image_0'))
TRANSFORMATIONS_PATH = 'transformations_ex3.npy'

# k_object = gtsam.Cal3_S2(718.856, 718.856, 0, 607.1928, 185.2157, 0.53716572)
# k_object = gtsam.Cal3_S2Stereo(718.856, 718.856, 0, 607.1928, 185.2157, 0.53716572)


k_object = gtsam.Cal3_S2Stereo(K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2], BASE_LINE_SIGN * M2[0, 3])


# # Select keyframes based on a criterion (e.g., every 10 frames)
# # keyframe_indices = list(range(0, len(transformations), 10))
# keyframe_indices = extract_keyframes(tracking_db)
# # Initialize the factor graph and initial estimates
# graph = gtsam.NonlinearFactorGraph()
# initial_estimate = gtsam.Values()
#
# # Add camera poses and landmark estimates to the initial estimate
# for i, frame_idx in enumerate(keyframe_indices):
#     pose_symbol = symbol('c', i)
#     camera_matrix = transformations[frame_idx]
#     rotation = camera_matrix[:3, :3]
#     translation = camera_matrix[:3, 3]
#     inverse_rotation = rotation.T
#     inverse_translation = -inverse_rotation @ translation
#     pose = gtsam.Pose3(gtsam.Rot3(inverse_rotation), gtsam.Point3(inverse_translation))
#     initial_estimate.insert(pose_symbol, pose)
#
# # Add reprojection factors
# for track_id in db.trackId_to_frames:
#     frames = db.trackId_to_frames[track_id]
#     for frame in frames:
#         if frame in keyframe_indices:
#             frame_idx = keyframe_indices.index(frame)
#             pose_symbol = symbol('c', frame_idx)
#             link = db.link(frame, track_id)
#             stereo_point2d = gtsam.StereoPoint2(link.x_left, link.x_right, link.y)
#             graph.add(gtsam.GenericStereoFactor3D(stereo_point2d, gtsam.noiseModel.Isotropic.Sigma(3, 1.0), pose_symbol,
#                                                   symbol('p', track_id), K))
fig_num = -1


def get_fig_num():
    """Get the next figure number."""
    global fig_num
    fig_num += 1
    return fig_num


def get_relevant_tracks_in_keyframes(db, first_frame_idx, last_frame_idx):
    """get all the tracks that participate in more then one frame between the first and last frame."""
    all_tracks = set()
    frame_set = set(db.tracks(first_frame_idx))
    for i in range(first_frame_idx + 1, last_frame_idx + 1):
        current_frame_id = i
        current_frame_tracks = set(db.tracks(current_frame_id))
        intersection = frame_set.intersection(current_frame_tracks)
        all_tracks.update(intersection)
        frame_set = current_frame_tracks
    return all_tracks


def calculate_transformations(db: TrackingDB, first_frame_idx, last_frame_idx):
    """Calculate the realtive transformations between the first and every frame between
        the first and the last frame."""
    transformations = dict()
    transformations[0] = np.vstack((M1, np.array([0, 0, 0, 1])))

    for i in range(1, last_frame_idx - first_frame_idx + 1):
        current_frame_id = i
        prev_frame_id = current_frame_id - 1
        last_frame_transform = transformations[prev_frame_id]

        # get the relevant tracks
        common_tracks = list(set(db.tracks(current_frame_id)).intersection(
            set(db.tracks(prev_frame_id))))
        links_last_frame = [db.link(prev_frame_id, track_id) for track_id in common_tracks]
        links_first_frame = [db.link(current_frame_id, track_id) for track_id in common_tracks]

        # calculate the transformation between the two frames
        # traingulate the links
        triangulated_links = triangulate_last_frame(db, P, Q, links_last_frame)
        if len(triangulated_links) < 4:
            raise Exception("Not enough points to triangulate and perform PnP")
        # calculate the transformation
        diff_coeff = np.zeros((5, 1))
        links_first_frame = np.array(links_first_frame)
        img_points = np.array([(link.x_left, link.y) for link in links_first_frame])
        # links_current_frame = np.array(links_current_frame)
        success, rotation_vector, translation_vector = cv2.solvePnP(triangulated_links, img_points, K,
                                                                    distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)

        inv_t = rodriguez_to_mat(rotation_vector, translation_vector) if success else None
        if inv_t is None:
            raise Exception("PnP failed")
        new_trans = get_inverse(inv_t)
        transformations[current_frame_id] = get_inverse(new_trans @ last_frame_transform)

    # last_transform = transformations[last_frame_id]
    # global_transform = new_trans @ np.vstack((last_transform, np.array([0, 0, 0, 1])))
    # transformations[current_frame_id] = global_transform
    return transformations


# def calculate_transformation_a_to_b(first_frame_idx, last_frame_idx, transformations):
#     new_transforms = [M1]
#     t_1 = np.vstack((transformations[first_frame_idx, :, :], np.array([0, 0, 0, 1])))
#     # inv_first = np.linalg.inv(t_1)
#     rotation = transformations[first_frame_idx, :3, :3]
#     translation = transformations[first_frame_idx, :3, 3]
#     inverse_rotation = rotation.T
#     inverse_translation = -inverse_rotation @ translation
#     inv_first = np.hstack((inverse_rotation, inverse_translation.reshape(-1, 1)))
#     inv_first = np.vstack((inv_first, np.array([0, 0, 0, 1])))
#     for i in range(first_frame_idx, last_frame_idx):
#         # new_transform = inv_first @ (np.vstack((transformations[i + 1, :, :], np.array([0, 0, 0, 1]))))
#         rotation = transformations[i + 1, :3, :3]
#         translation = transformations[i + 1, :3, 3]
#         inverse_rotation = rotation.T
#         inverse_translation = -inverse_rotation @ translation
#         new_transform = np.hstack((inverse_rotation, inverse_translation.reshape(-1, 1)))
#         new_transform = inv_first @ (np.vstack((transformations[i + 1, :, :], np.array([0, 0, 0, 1]))))
#         # new_transform = inv_first @ (np.vstack((new_transform, np.array([0, 0, 0, 1]))))
#         # new_transform = t_1 @ (np.vstack((new_transform, np.array([0, 0, 0, 1]))))
#         # print(f"new transform {new_transform[:, :]}")
#         new_transforms.append(new_transform[:, :])
#     return new_transforms


# def create_factors_between_keyframes(graph, first_frame_idx, last_frame_idx, db, k_matrix, initial_estimate,
#                                      transformations):
#
#     gtsam_frames = dict()
#     camera_symbols = dict()
#     # relative_transformations = calculate_transformation_a_to_b(first_frame_idx, last_frame_idx, transformations)
#     relative_transformations = transformations[first_frame_idx:last_frame_idx + 1]
#     first = relative_transformations[0]
#     inverse_first = get_inverse(first)
#     first_pose = gtsam.Pose3(gtsam.Rot3(inverse_first[:3, :3]), gtsam.Point3(inverse_first[:3, 3]))
#     # print(f" first frame {first_frame_idx} last frame {last_frame_idx} relevant {len(relevant_tracks)}")
#     for i in range(last_frame_idx + 1 - first_frame_idx):
#         frame_id = first_frame_idx + i
#         pose_symbol = gtsam.symbol('c', frame_id)
#         camera_matrix = relative_transformations[i]
#
#         camera_inverse = get_inverse(camera_matrix)
#         # pose_i = gtsam.Pose3(gtsam.Rot3(camera_inverse[:3, :3]), gtsam.Point3(camera_inverse[:3, 3]))
#         # relative_pose = first_pose.between(pose_i)
#         final_mat = np.vstack((first,np.array([0,0,0,1]))) @ camera_inverse
#         # final_mat = np.vstack((first,np.array([0,0,0,1]))) @ camera_inverse
#
#         # translated = inverse_first @ np.vstack((camera_matrix, np.array([0, 0, 0, 1])))
#         # camera_inverse = get_inverse(translated)
#         # final_mat = camera_inverse
#
#         # divide by the last element
#         # for j in range(4):
#         #     final_mat[:, j] /= final_mat[3, j]
#         # print(f"pose {final_mat}")
#         rotation = final_mat[:3, :3]
#         translation = final_mat[:3, 3]
#
#
#         camera_matrix = calculate_transformations(db,first_frame_idx,last_frame_idx)
#         inv = get_inverse(camera_matrix)
#         rotation = inv[:3, :3]
#         translation = inv[:3, 3]
#
#
#         # pose = gtsam.Pose3(gtsam.Rot3(inverse_rotation), gtsam.Point3(inverse_translation))
#         if i == 0:
#             # pose = first_pose
#             pose = gtsam.Pose3()
#             graph.add(gtsam.PriorFactorPose3(pose_symbol, pose, gtsam.noiseModel.Isotropic.Sigma(6, 1.0)))
#         else:
#             pose = gtsam.Pose3(gtsam.Rot3(rotation), gtsam.Point3(translation))
#             # pose = relative_pose
#         initial_estimate.insert(pose_symbol, pose)
#         # Create the stereo camera
#         # gtsam_frame = gtsam.StereoCamera(pose, k_matrix)
#         gtsam_frame = gtsam.StereoCamera(pose, k_matrix)
#         gtsam_frames[frame_id] = gtsam_frame
#         camera_symbols[frame_id] = pose_symbol
#         # if i == 0:
#         #     graph.add(gtsam.PriorFactorPose3(pose_symbol, pose, gtsam.noiseModel.Isotropic.Sigma(6, 1.0)))
#
#         frame_tracks = sorted(db.tracks(frame_id))
#         for track_id in frame_tracks:
#             location_symbol = gtsam.symbol('l', track_id)
#             link = db.link(frame_id, track_id)
#
#             track_last_frame = db.last_frame_of_track(track_id)
#             track_last_frame = min(track_last_frame, last_frame_idx)
#             # last_frame_link = db.link(track_last_frame, track_id)
#             stereo_point2d = link.x_left, link.x_right, link.y
#             # gtsam_frame = gtsam_frames[track_last_frame]
#             # reference_triangulated_point = gtsam_frame.backproject(
#             #     gtsam.StereoPoint2(stereo_point2d[0], stereo_point2d[1], stereo_point2d[2]))
#
#             # initial_estimate.insert(location_symbol, reference_triangulated_point)
#
#             if frame_id == track_last_frame or frame_id == last_frame_idx:
#                 # triangulate the point in the last frame
#                 gtsam_frame = gtsam_frames[frame_id]
#                 reference_triangulated_point = gtsam_frame.backproject(
#                     gtsam.StereoPoint2(stereo_point2d[0], stereo_point2d[1], stereo_point2d[2]))
#                 # triangulated_point = linear_least_squares_triangulation(P, Q, (stereo_point2d[0], stereo_point2d[2]),
#                 #                                                         (stereo_point2d[1], stereo_point2d[2]))
#                 # triangulated_point = triangulated_point.reshape(-1, 1)
#                 # # print(f"triangulated shape: {triangulated_point.shape}")
#                 # triangulated_point = final_mat @ np.vstack((triangulated_point, np.array([1])))
#                 # # print(f"triangulated point {triangulated_point} reference {reference_triangulated_point}")
#                 # triangulated_point /= triangulated_point[3]
#                 # triangulated_point = triangulated_point[:3]
#                 # diff = np.linalg.norm(triangulated_point - reference_triangulated_point.reshape(-1, 1))
#                 # print(f"triangulated point {triangulated_point.reshape(1,-1)} reference {reference_triangulated_point}, diff {diff}")
#                 # print(f"diff {diff}")
#
#                 initial_estimate.insert(location_symbol, reference_triangulated_point)
#
#             # Create the factor
#             sigma = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
#
#             graph.add(
#                 gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(link.x_left, link.x_right, link.y), sigma, pose_symbol,
#                                             location_symbol,
#                                             k_object))
#
#             # graph.add(fac)
#
#     # for track_id in relevant_tracks:
#     #     frames = db.frames(track_id)
#     #     frames = sorted(frames)
#     #     frames = [frame for frame in frames if first_frame_idx <= frame <= last_frame_idx]
#     #     # track_last_frame = frames[-1]
#     #     # link = db.link(track_last_frame, track_id)
#     #     # stereo_point2d = link.x_left, link.x_right, link.y
#     #
#     #     # triangulate the point in the last frame
#     #     # gtsam_last_frame = gtsam_frames[track_last_frame]
#     #     # reference_triangulated_point = gtsam_last_frame.backproject(
#     #     #     gtsam.StereoPoint2(stereo_point2d[0], stereo_point2d[1], stereo_point2d[2]))
#     #
#     #     # initial_estimate.insert(location_symbol, reference_triangulated_point)
#     #
#     #     location_symbol = gtsam.symbol('l', track_id)
#     #     for j in range(len(frames)):
#     #         frame = frames[j]
#     #         link = db.link(frame, track_id)
#     #         if j == len(frames) - 1:
#     #             stereo_point2d = link.x_left, link.x_right, link.y
#     #             #
#     #             # # Create the stereo camera
#     #             gt_frame_2 = gtsam_frames[frame]
#     #
#     #             # # # Triangulate the point in the last frame
#     #             projected_point_estimated = gt_frame_2.backproject(
#     #                 gtsam.StereoPoint2(stereo_point2d[0], stereo_point2d[1], stereo_point2d[2]))
#     #
#     #             initial_estimate.insert(location_symbol, gtsam.Point3(projected_point_estimated))
#     #
#     #         # Create the factor
#     #         sigma = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
#     #         c = camera_symbols[frame]
#     #         fac = gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(link.x_left, link.x_right, link.y), sigma, c,
#     #                                           location_symbol,
#     #                                           k_object)
#     #         graph.add(fac)
#     return graph, initial_estimate

# insert the initial estimation of the triangulated point

# todo: implement it!
# def calculate_all_relative_transformations_to_gtsam(all_transformations, first_frame, last_frame):
#     new_trans = []
#     r_first = all_transformations[first_frame][:3, :3]
#     t_first = all_transformations[first_frame][:3, 3]
#     inverse = get_inverse(all_transformations[first_frame])
#     for trans in all_transformations[first_frame: last_frame + 1]:
#         t_inv = get_inverse(inverse @ np.vstack((trans, np.array([0, 0, 0, 1]))))
#         # t_inv = inverse @ np.vstack((trans, np.array([0, 0, 0, 1])))
#         # rel_transformation = np.hstack((new_r, new_t.reshape(-1, 1)))
#         # rel_transformation = get_inverse(rel_transformation)
#         new_trans.append(t_inv)
#
#     return new_trans


def create_factor_graph(first_frame_idx, last_frame_idx, db):
    """ Create a factor graph for the given frames."""
    gtsam_frames = dict()
    camera_symbols = dict()
    relevant_tracks = set()
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # calculate the relative transformations
    relative_transformations = calculate_transformations(db, first_frame_idx, last_frame_idx)
    pose_sigma = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]))

    # add the cameras poses to the graph
    for i in range(last_frame_idx + 1 - first_frame_idx):
        frame_id = first_frame_idx + i
        pose_symbol = gtsam.symbol('c', frame_id)
        camera_pose = relative_transformations[i]

        # add the first camera pose prior
        if i == 0:
            pose = gtsam.Pose3()
            graph.add(gtsam.PriorFactorPose3(pose_symbol, pose, pose_sigma))
            initial_estimate.insert(pose_symbol, pose)
            gtsam_frame = gtsam.StereoCamera(pose, k_object)
            gtsam_frames[frame_id] = gtsam_frame
            camera_symbols[frame_id] = pose_symbol
            continue

        else:
            last_tracks = set(db.tracks(frame_id - 1))
            this_frame_tracks = set(db.tracks(frame_id))
            intersection = last_tracks.intersection(this_frame_tracks)
            relevant_tracks.update(intersection)
            rotation = camera_pose[:3, :3]
            translation = camera_pose[:3, 3]
            pose = gtsam.Pose3(gtsam.Rot3(rotation), gtsam.Point3(translation))
            initial_estimate.insert(pose_symbol, pose)

        # Create the stereo camera
        gtsam_frame = gtsam.StereoCamera(pose, k_object)
        gtsam_frames[frame_id] = gtsam_frame
        camera_symbols[frame_id] = pose_symbol

    relevant_tracks = list(get_relevant_tracks_in_keyframes(db, first_frame_idx, last_frame_idx))

    # add the stereo factors to the graph
    for track_id in relevant_tracks:
        tracks_frames = sorted(db.frames(track_id), reverse=True)
        track_last_frame = min(db.last_frame_of_track(track_id), last_frame_idx)
        location_symbol = gtsam.symbol('l', track_id)
        for frame_id in tracks_frames:
            if frame_id > last_frame_idx or frame_id < first_frame_idx:
                continue
            else:
                pose_symbol = camera_symbols[frame_id]
                link = db.link(frame_id, track_id)
                stereo_point2d = link.x_left, link.x_right, link.y
                assert stereo_point2d[0] > stereo_point2d[1]

                if frame_id == track_last_frame or frame_id == last_frame_idx:
                    # triangulate the point in the last frame and add measurement

                    gtsam_frame = gtsam_frames[frame_id]
                    reference_triangulated_point = gtsam_frame.backproject(
                        gtsam.StereoPoint2(stereo_point2d[0], stereo_point2d[1], stereo_point2d[2]))

                    assert reference_triangulated_point[2] > 0

                    initial_estimate.insert(location_symbol, reference_triangulated_point)

                # Create the factor
                # n = np.array([0.1, 0.1, 0.5])*(track_last_frame-frame_id+1) + np.array([1, 1, 1])
                # sigma = gtsam.noiseModel.Diagonal.Sigmas(n)
                sigma = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
                graph.add(
                    gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(link.x_left, link.x_right, link.y), sigma,
                                                pose_symbol,
                                                location_symbol,
                                                k_object))

    return graph, initial_estimate, camera_symbols, gtsam_frames


def get_inverse(first):
    """ Get the inverse of a transformation matrix."""
    inverse_rotation = first[:3, :3].T
    inverse_translation = -inverse_rotation @ first[:3, 3]
    inverse_first = np.hstack((inverse_rotation, inverse_translation.reshape(-1, 1)))
    inverse_first = np.vstack((inverse_first, np.array([0, 0, 0, 1])))
    return inverse_first


# def translate_later_t_to_older_t(later_t, older_t):
#     return older_t @ get_inverse(later_t)


def get_factor_symbols(factor):
    """ Get the symbols of the factor."""
    camera_symbol = factor.keys()[0]
    location_symbol = factor.keys()[1]
    return camera_symbol, location_symbol


def get_factor_point(factor, gt_values):
    """ Get the point of the factor."""
    factor_landmark = get_factor_symbols(factor)[1]
    return gt_values.atPoint3(factor_landmark)


def get_negative_z_points(result, graph):
    """ Remove points with negative z values, from the graph and the result. """
    graph_size = graph.size()
    values_to_remove = []
    factors_to_remove = []
    for i in range(graph_size):
        factor = graph.at(i)
        if isinstance(factor, gtsam.GenericStereoFactor3D):
            point = get_factor_point(factor, result)
            if point[2] < 0:
                camera_symbol, track_symbol = get_factor_symbols(factor)
                values_to_remove.append(track_symbol)
                factors_to_remove.append(i)

    for ind in factors_to_remove:
        # if ind == len(factors_to_remove )
        graph.remove(ind)
    values_to_remove = list(set(values_to_remove))
    for value_to_remove in values_to_remove:
        result.erase(value_to_remove)
    return result, graph, len(values_to_remove) > 0


def optimize_graph(graph, initial):
    """ Optimize the graph using the Levenberg-Marquardt optimizer in iterative manner.
    Remove points with negative z values from the graph and the result after optimization until there are no such points."""
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()
    # return graph, result
    result, new_graph, removed = get_negative_z_points(result, graph)
    if removed:
        print("the results keys before filtering", len(result.keys()))
    while removed:
        optimizer = gtsam.LevenbergMarquardtOptimizer(new_graph, result)
        result = optimizer.optimize()
        result, new_graph, removed = get_negative_z_points(result, new_graph)
    return new_graph, result


def print_q_5_4(bundles, cameras_locations, cameras_locations2, distance, keyframes_indices):
    """ Print the results and plots of the 5th question of the 4th assignment."""
    first_cam_last_bundle = cameras_locations[-2]
    print(f"first camera of last bundle: {first_cam_last_bundle}")
    num_bundles = len(bundles)
    last_bundle = bundles[num_bundles - 1]
    last_bundle_result = last_bundle['result']
    graph = last_bundle['graph']
    camera_symbol = last_bundle['cameras_dict'][keyframes_indices[-1][0]]
    anchoring_factor = None
    num_factors = graph.size()
    for i in range(num_factors):
        factor = graph.at(i)
        if isinstance(factor, gtsam.PriorFactorPose3):
            anchoring_factor = factor
            break
    print(f"result error: {graph.error(last_bundle_result)}")
    print(f"initial error: {graph.error(last_bundle['initial'])}")
    anchoring_factor_error = anchoring_factor.error(last_bundle_result)
    print(f"Anchoring factor error: {anchoring_factor_error}")
    # print_graph_errors(graph, initial, key_frames, result)
    # Plot the optimized trajectory
    # worst_factor = print_biggest_error(graph, initial, result)
    # print(cameras_locations)
    # present the camera locations in 2D
    plt.figure(get_fig_num())
    plt.plot([x[0] for x in cameras_locations2], [x[2] for x in cameras_locations2], 'bo', label='ground truth',
             markersize=7)
    plt.plot([x[0] for x in cameras_locations], [x[2] for x in cameras_locations], 'ro', markersize=5)
    plt.title('Camera locations in 2D')
    plt.figure(get_fig_num())
    plt.plot([i for i in range(len(distance))], [distance[i] for i in range(len(distance))], 'bo', label='ground truth',
             markersize=1)
    plt.title('Camera distance from ground truth')


def present_error(camera_symbol, initial, result, location_symbol, worst_factor, title1='', title2=''):
    """ Present the error of the worst factor in the graph."""
    stereo_cam = gtsam.StereoCamera(initial.atPose3(camera_symbol), k_object)
    # initial_stereo_point = gtsam.StereoPoint2(initial.atPoint3(location_symbol)[0],
    #                                           initial.atPoint3(location_symbol)[1],
    #                                           initial.atPoint3(location_symbol)[2])
    measurement = worst_factor.measured()
    location = initial.atPoint3(location_symbol)
    projected_point = stereo_cam.project(location)
    projected_left_cam = projected_point.uL(), projected_point.v()
    projected_right_cam = projected_point.uR(), projected_point.v()
    camera_number = int(gtsam.DefaultKeyFormatter(camera_symbol)[1:])
    img_left, img_right = read_images(camera_number)
    measurement_left = measurement.uL(), measurement.v()
    measurement_right = measurement.uR(), measurement.v()
    distance_left = round(np.linalg.norm(np.array(measurement_left) - np.array(projected_left_cam)), 2)
    distance_right = round(np.linalg.norm(np.array(measurement_right) - np.array(projected_right_cam)), 2)
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))
    axs[0, 0].title.set_text(f"Left camera {camera_number}, distance: {distance_left} {title1}")
    axs[0, 0].imshow(img_left, cmap='gray')
    axs[0, 0].scatter(*projected_left_cam, c='r', label='Projection')
    axs[0, 0].scatter(*measurement_left, c='g', label='Measurement')
    axs[0, 0].legend()
    axs[0, 1].title.set_text(f"Right camera {camera_number}, distance: {distance_right} {title1}")
    axs[0, 1].imshow(img_right, cmap='gray')
    axs[0, 1].scatter(*projected_right_cam, c='r', label='Projection')
    axs[0, 1].scatter(*measurement_right, c='g', label='Measurement')
    axs[0, 1].legend()

    world_point = stereo_cam.backproject(measurement)
    print(f"point in the global coordinate system before optimization: {world_point}")
    print(f"estimated location by the camera: {location}")
    distance_in_meters = np.linalg.norm(world_point - np.array(location))
    print(f"Distance in meters: {distance_in_meters}")

    stereo_cam = gtsam.StereoCamera(result.atPose3(camera_symbol), k_object)
    location = result.atPoint3(location_symbol)
    projected_point = stereo_cam.project(location)
    projected_left_cam = projected_point.uL(), projected_point.v()
    projected_right_cam = projected_point.uR(), projected_point.v()
    camera_number = int(gtsam.DefaultKeyFormatter(camera_symbol)[1:])
    img_left, img_right = read_images(camera_number)
    measurement_left = measurement.uL(), measurement.v()
    measurement_right = measurement.uR(), measurement.v()
    distance_left = round(np.linalg.norm(np.array(measurement_left) - np.array(projected_left_cam)), 2)
    distance_right = round(np.linalg.norm(np.array(measurement_right) - np.array(projected_right_cam)), 2)
    axs[1, 0].title.set_text(f"Left camera {camera_number}, distance: {distance_left} {title2}")
    axs[1, 0].imshow(img_left, cmap='gray')
    axs[1, 0].scatter(*projected_left_cam, c='r', label='Projection')
    axs[1, 0].scatter(*measurement_left, c='g', label='Measurement')
    axs[1, 0].legend()
    axs[1, 1].title.set_text(f"Right camera {camera_number}, distance: {distance_right} {title2}")
    axs[1, 1].imshow(img_right, cmap='gray')
    axs[1, 1].scatter(*projected_right_cam, c='r', label='Projection')
    axs[1, 1].scatter(*measurement_right, c='g', label='Measurement')
    axs[1, 1].legend()

    world_point = stereo_cam.backproject(measurement)
    print(f"point in the global coordinate system after optimization: {world_point}")
    print(f"estimated location by the camera: {location}")
    distance_in_meters = np.linalg.norm(world_point - np.array(location))
    print(f"Distance in meters: {distance_in_meters}")


# def project_point_to_cam(initial, result, worst_factor):
#     camera_symbol = worst_factor.keys()[0]
#     location_symbol = worst_factor.keys()[1]
#     print(f"Initial triangulated point: {initial.atPoint3(location_symbol)}")
#     print(f"Final triangulated point: {result.atPoint3(location_symbol)}")
#     print(f"Initial camera pose: {initial.atPose3(camera_symbol)}")
#     print(f"Final camera pose: {result.atPose3(camera_symbol)}")
#
#     stereo_cam = gtsam.StereoCamera(initial.atPose3(camera_symbol), k_object)
#     initial_stereo_point = gtsam.StereoPoint2(initial.atPoint3(location_symbol)[0],
#                                               initial.atPoint3(location_symbol)[1],
#                                               initial.atPoint3(location_symbol)[2])
#     initial_projected_point = stereo_cam.project(initial_stereo_point)
#     print(f"Initial estimation projected to camera: {stereo_cam.project(initial_stereo_point)}")
#     after_opt_stereo_point = gtsam.StereoPoint2(result.atPoint3(location_symbol)[0],
#                                                 result.atPoint3(location_symbol)[1],
#                                                 result.atPoint3(location_symbol)[2])
#     return after_opt_stereo_point, stereo_cam


def print_biggest_error(graph: gtsam.NonlinearFactorGraph, initial: gtsam.Values, result: gtsam.Values):
    """Prints the biggest error in the graph before and after optimization."""
    worst_factor = None
    num_factors = graph.size()
    for i in range(num_factors):
        factor = graph.at(i)
        if isinstance(factor, gtsam.GenericStereoFactor3D):
            if worst_factor is None or factor.error(initial) > worst_factor.error(initial):
                worst_factor = factor
    print(f"Worst factor error: {worst_factor.error(initial)}")
    print(f"Worst factor error after optimization: {worst_factor.error(result)}")
    return worst_factor


def print_graph_errors(graph, initial, key_frames, result):
    """Prints the total error of the graph before and after optimization."""
    initial_error = graph.error(initial)
    final_error = graph.error(result)
    num_factors = graph.size()
    print(f"first frame {key_frames[0]} last frame {key_frames[1]} ")
    print(f"Initial total error: {initial_error}")
    print(f"Final total error: {final_error}")
    print(f"Number of factors: {num_factors}")
    print(f"Average factor error before optimization: {initial_error / num_factors}")
    print(f"Average factor error after optimization: {final_error / num_factors}")


#     keyframe_positions = [result.atPose3(gtsam.symbol('c', i[0])).translation() for i in indices]
#     x_positions = [pose.x() for pose in keyframe_positions]
#     z_positions = [pose.z() for pose in keyframe_positions]
#
#     # Plot the keyframe positions
#     ax.plot(x_positions, z_positions, 'ro-', label='Keyframe Trajectory')
#
#     # If you have 3D landmarks, plot them as well
#     landmark_keys = [key for key in result.keys() if gtsam.symbolChr(key) == 'p']
#     landmark_positions = [result.atPoint3(key) for key in landmark_keys]
#     landmark_x = [point.x() for point in landmark_positions]
#     landmark_z = [point.z() for point in landmark_positions]
#
#     ax.scatter(landmark_x, landmark_z, c='b', marker='x', label='3D Landmarks')
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Z')
#     ax.set_title('View from Above')
#     ax.legend()
#     ax.grid(True)
#
#


# def transform_to_gtsam_pose(frame_idx, transformations, frames):
#     reference_transform = transformations[frames[0]]
#     transform = transformations[frame_idx]
#     relavite = translate_later_t_to_older_t(transform, reference_transform)
#     return relavite


# def add_camera_poses(graph, transformations, frames, initial):
#     camera_symbols = dict()
#     frame_objects = dict()
#
#     for i, frame in enumerate(frames):
#         camera_symbol = gtsam.symbol('c', frame)
#         camera_symbols[frame] = camera_symbol
#
#         pose_matrix = transform_to_gtsam_pose(frame, transformations, frames)
#         pose = gtsam.Pose3(gtsam.Rot3(pose_matrix[:3, :3]), gtsam.Point3(pose_matrix[:3, 3]))
#         gt_frame = gtsam.StereoCamera(pose, k_object)
#         frame_objects[frame] = gt_frame
#         initial.insert(camera_symbol, pose)
#         if i == 0:
#             graph.add(gtsam.PriorFactorPose3(camera_symbol, pose, gtsam.noiseModel.Isotropic.Sigma(6, 1.0)))
#     return camera_symbols, frame_objects


# def optimize_track(db, track_id, transformations, fist_frame_idx, last_frame_idx):
#     graph = gtsam.NonlinearFactorGraph()
#     initial = gtsam.Values()
#
#     if fist_frame_idx == last_frame_idx:
#         frames = db.frames(track_id)
#     else:
#         frames = sorted([frame for frame in db.frames(track_id) if frame >= fist_frame_idx and frame <= last_frame_idx])
#
#     camera_symbols, frame_objects = add_camera_poses(graph, transformations, frames, initial)
#
#     track_last_frame_index = frames[-1]
#     last_frame = frame_objects[track_last_frame_index]
#     location_symbol = gtsam.symbol('l', track_id)
#     last_frame_link = db.link(track_last_frame_index, track_id)
#
#     location = last_frame.backproject(
#         gtsam.StereoPoint2(last_frame_link.x_left, last_frame_link.x_right, last_frame_link.y))
#     initial.insert(location_symbol, location)
#     for frame in frames:
#         link = db.link(frame, track_id)
#         pose_symbol = camera_symbols[frame]
#         sigma = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
#         factor = gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(link.x_left, link.x_right, link.y), sigma, pose_symbol,
#                                              location_symbol,
#                                              k_object)
#         graph.add(factor)
#
#     # Create the optimizer
#     optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
#     result = optimizer.optimize()
#     if graph.error(result) > 0.5:
#         print(f"error after optimization {graph.error(result)}, track {track_id} frames {len(frames)} ")


# def find_bad_tracks(first_frame_idx, last_frame_idx, db, k_matrix,
#                     transformations):
#     # all_tracks_in_keyframes = set()
#     gtsam_frames = dict()
#     camera_symbols = dict()
#     relevant_tracks = get_relevant_tracks_in_keyframes(db, first_frame_idx, last_frame_idx)
#     # relative_transformations = calculate_transformation_a_to_b(first_frame_idx, last_frame_idx, transformations)
#     relative_transformations = transformations[first_frame_idx:last_frame_idx + 1]
#     first = relative_transformations[0]
#     inverse_first = get_inverse(first)
#     first_pose = gtsam.Pose3(gtsam.Rot3(inverse_first[:3, :3]), gtsam.Point3(inverse_first[:3, 3]))
#     # print(f" first frame {first_frame_idx} last frame {last_frame_idx} relevant {len(relevant_tracks)}")
#     for i in range(last_frame_idx + 1 - first_frame_idx):
#         frame_id = first_frame_idx + i
#         pose_symbol = gtsam.symbol('c', frame_id)
#         camera_matrix = relative_transformations[i]
#         camera_inverse = get_inverse(camera_matrix)
#         pose_i = gtsam.Pose3(gtsam.Rot3(camera_inverse[:3, :3]), gtsam.Point3(camera_inverse[:3, 3]))
#         relative_pose = first_pose.between(pose_i)
#         final_mat = inverse_first @ camera_inverse
#
#         # translated = inverse_first @ np.vstack((camera_matrix, np.array([0, 0, 0, 1])))
#         # camera_inverse = get_inverse(translated)
#         # final_mat = camera_inverse
#
#         # divide by the last element
#         # for j in range(4):
#         #     final_mat[:, j] /= final_mat[3, j]
#         # print(f"pose {final_mat}")
#         rotation = final_mat[:3, :3]
#         translation = final_mat[:3, 3]
#         # pose = gtsam.Pose3(gtsam.Rot3(inverse_rotation), gtsam.Point3(inverse_translation))
#         # if i == 0:
#         #     pose = first_pose
#         #     graph.add(gtsam.PriorFactorPose3(pose_symbol, pose, gtsam.noiseModel.Isotropic.Sigma(6, 1.0)))
#         # else:
#         #     pose = gtsam.Pose3(gtsam.Rot3(rotation), gtsam.Point3(translation))
#         #     # pose = relative_pose
#         # initial_estimate.insert(pose_symbol, pose)
#         # Create the stereo camera
#         # gtsam_frame = gtsam.StereoCamera(pose, k_matrix)
#         pose = gtsam.Pose3(gtsam.Rot3(rotation), gtsam.Point3(translation))
#         gtsam_frame = gtsam.StereoCamera(pose, k_matrix)
#         gtsam_frames[frame_id] = gtsam_frame
#         camera_symbols[frame_id] = pose_symbol
#         # if i == 0:
#         #     graph.add(gtsam.PriorFactorPose3(pose_symbol, pose, gtsam.noiseModel.Isotropic.Sigma(6, 1.0)))
#
#     # frame_tracks = db.tracks(frame_id)
#     g = gtsam.NonlinearFactorGraph()
#     est = gtsam.Values()
#     inserted_pose = []
#     for track_id in relevant_tracks:
#         track_last_frame = min(db.last_frame_of_track(track_id), last_frame_idx)
#         pose_symbol = camera_symbols[track_last_frame]
#         first_pose = gtsam_frames[first_frame_idx].pose()
#         first_symbol = camera_symbols[first_frame_idx]
#         g.add(gtsam.PriorFactorPose3(first_symbol, first_pose, gtsam.noiseModel.Isotropic.Sigma(6, 1.0)))
#         gtsam_frame = gtsam_frames[track_last_frame]
#         pose = gtsam_frame.pose()
#         # est.insert(pose_symbol, pose)
#
#         link = db.link(track_last_frame, track_id)
#         stereo_point2d = link.x_left, link.x_right, link.y
#         triangulated_point = gtsam_frame.backproject(
#             gtsam.StereoPoint2(stereo_point2d[0], stereo_point2d[1], stereo_point2d[2]))
#         est.insert(gtsam.symbol('l', track_id), triangulated_point)
#
#         location_symbol = gtsam.symbol('l', track_id)
#         for frame_id in sorted(db.frames(track_id)):
#
#             if frame_id < first_frame_idx or frame_id > last_frame_idx:
#                 continue
#
#             pose_symbol = camera_symbols[frame_id]
#             gtsam_frame = gtsam_frames[frame_id]
#             if pose_symbol not in inserted_pose:
#                 est.insert(pose_symbol, gtsam_frame.pose())
#                 inserted_pose.append(pose_symbol)
#             # est.insert(pose_symbol, gtsam_frame.pose())
#             link = db.link(frame_id, track_id)
#             stereo_point2d = link.x_left, link.x_right, link.y
#
#             sigma = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
#
#             fac = gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(link.x_left, link.x_right, link.y), sigma, pose_symbol,
#                                               location_symbol,
#                                               k_object)
#             g.add(fac)
#         # if 0 not in db.frames(track_id) or :
#         #     est.insert(first_symbol, first_pose)
#
#     error = g.error(est)
#     optimizer = gtsam.LevenbergMarquardtOptimizer(g, est)
#     result = optimizer.optimize()
#     initial_error = g.error(est)
#     final_error = g.error(result)
#
#     if final_error > 20:
#         print(f"track {error} error {final_error} track length {len(db.frames(track_id))}")
#
#         # for fac in g.keys():
#         #     print(fac.error(est))


def calc_locations_angle(t1, t2):
    """ Calculate the angle between two cameras in the x-y plane for keyframe selection"""
    cam1_rotation = t1[:3, :3]
    cam1_translation = t1[:3, 3]
    cam1_position = -cam1_rotation.T @ cam1_translation
    cam1_position = cam1_position[:2]

    cam2_rotation = t2[:3, :3]
    cam2_translation = t2[:3, 3]
    cam2_position = -cam2_rotation.T @ cam2_translation
    cam2_position = cam2_position[:2]
    x = max(np.abs(cam2_position[1] - cam1_position[1]), np.abs(cam2_position[0] - cam1_position[0]))
    y = min(np.abs(cam2_position[1] - cam1_position[1]), np.abs(cam2_position[0] - cam1_position[0]))
    angle = np.arctan2(y, x)
    # angle = np.arctan2(np.abs(cam2_position[1] - cam1_position[1]), np.abs(cam2_position[0] - cam1_position[0]))

    return angle


def extract_keyframes(db: TrackingDB, transformations):
    """ Extract keyframes from the database"""
    keyFrames = []  # all keyframes start with zero
    frames = db.all_frames()
    num_frames = len(frames)
    i = 0
    minimum_gap = 5
    max_dist = 5.0
    track_losing_factor = 0.3
    max_gap = 10

    theta_max = 20

    while i < num_frames - 1:
        t_initial = transformations[i]
        t1 = t_initial
        old_tracks = set(db.tracks(i))
        start = min(i + minimum_gap, num_frames - 1)
        dist = 0
        # accumalate_angle = 0
        for j in range(start, min(i + max_gap, num_frames)):
            t2 = transformations[j]
            dist = calculate_distance_between_keyframes(t_initial, t2)
            new_tracks = set(db.tracks(j))
            common_tracks = old_tracks.intersection(new_tracks)
            tracks_ratio = len(common_tracks) / len(old_tracks)
            old_tracks = new_tracks
            accumalate_angle = (np.abs(calc_locations_angle(t_initial, t2)) * 180 / np.pi)
            t1 = t2
            # print(f"angle {accumalate_angle} dist {dist} common tracks {len(common_tracks)}, frame {j}, i {i}")

            if tracks_ratio < track_losing_factor or j == i + max_gap - 1 \
                    or j == num_frames - 1 or dist > max_dist or accumalate_angle > theta_max:
                keyFrames.append((i, j))
                i = j
                break

        # Update i to j+1 if no keyframe was added in the inner loop
        if j == min(i + max_gap - 1, num_frames - 1):
            i = j + 1

    return keyFrames


def calculate_distance_between_keyframes(t1, t2):
    """ Calculate the distance in meters between two cameras for keyframe selection"""
    cam1_rotation = t1[:3, :3]
    cam1_translation = t1[:3, 3]
    cam1_position = -cam1_rotation.T @ cam1_translation

    cam2_rotation = t2[:3, :3]
    cam2_translation = t2[:3, 3]
    cam2_position = -cam2_rotation.T @ cam2_translation

    return np.linalg.norm(cam1_position - cam2_position)


def q_5_1(tracking_db: TrackingDB):
    """ Question 5.1"""
    def find_random_track_of_length(tracking_db: TrackingDB, length: int) -> Optional[int]:
        eligible_tracks = [trackId for trackId, frames in tracking_db.trackId_to_frames.items() if
                           len(frames) >= length]

        if not eligible_tracks:
            return None
        return random.choice(eligible_tracks)

    def calculate_plot_reprojection_err(find_random_track_of_length, plot_reprojection_errors, tracking_db, trackId):
        transformations = np.load(TRANSFORMATIONS_PATH)
        transformations = np.array(read_extrinsic_matrices())

        frames = tracking_db.frames(trackId)
        track_last_frame = tracking_db.last_frame_of_track(trackId)
        link = tracking_db.link(track_last_frame, trackId)
        stereo_point2d = link.x_left, link.x_right, link.y
        calculated_left_camera_matrix = transformations[track_last_frame]
        # calculated_left_camera_matrix = read_kth_camera(track_last_frame)

        # Calculate the inverse rotation
        rotation = calculated_left_camera_matrix[:3, :3]
        translation = calculated_left_camera_matrix[:3, 3]
        inverse_rotation = rotation.T

        # Calculate the inverse translation
        inverse_translation = -inverse_rotation @ translation

        # Create the left camera pose
        left_cam_pose = gtsam.Pose3(gtsam.Rot3(inverse_rotation), gtsam.Point3(inverse_translation))

        # Create the stereo camera
        gtsam_frame = gtsam.StereoCamera(left_cam_pose, k_object)

        # triangulate the point in the last frame
        reference_triangulated_point = gtsam_frame.backproject(
            gtsam.StereoPoint2(stereo_point2d[0], stereo_point2d[1], stereo_point2d[2]))

        l2_error = {}
        estimated_points = []
        sigma = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0]))
        sigma_6d = gtsam.noiseModel.Unit.Create(6)
        errors = {}

        # Create the symbol for the landmark
        l1 = gtsam.symbol('l', 1)

        # graph = gtsam.NonlinearFactorGraph()
        # c1 = gtsam.symbol('c', 1)
        # q1 = gtsam.symbol('q', 1)
        # pose_c1 = gtsam.Pose3()
        # loc_q1 = gtsam.Point3(0, 0, 0)
        # initialEstimate = gtsam.Values()
        # initialEstimate.insert(c1, pose_c1)
        # initialEstimate.insert(l1, loc_q1)
        for i in range(len(frames)):
            # extract the link x,y coordinates for the left and right camera
            frame = frames[i]
            print(f"trackId: {trackId}, frame: {frame}")
            link = tracking_db.link(frame, trackId)
            stereo_point2d_l = np.array([link.x_left, link.y])
            stereo_point2d_r = np.array([link.x_right, link.y])

            # Calculate the left camera pose
            calculated_left_camera_matrix = transformations[frame]
            rotation = calculated_left_camera_matrix[:3, :3]
            translation = calculated_left_camera_matrix[:3, 3]
            inverse_rotation = rotation.T

            # Calculate the inverse translation
            inverse_translation = -inverse_rotation @ translation

            # Create the left camera pose
            left_cam_pose = gtsam.Pose3(gtsam.Rot3(inverse_rotation), gtsam.Point3(inverse_translation))

            # Create the stereo camera
            gtsam_frame = gtsam.StereoCamera(left_cam_pose, k_object)

            # Triangulate the point in the last frame
            projected_point_estimated = gtsam_frame.project(reference_triangulated_point)

            # Estimate the the point feature coordinate in the left and right camera
            left_point_esti = np.array([projected_point_estimated.uL(), projected_point_estimated.v()])
            right_point_esti = np.array([projected_point_estimated.uR(), projected_point_estimated.v()])

            estimated_points.append((left_point_esti, right_point_esti))
            reprojection_error_left = np.linalg.norm(stereo_point2d_l - left_point_esti)
            reprojection_error_right = np.linalg.norm(stereo_point2d_r - right_point_esti)
            l2_error[int(track_last_frame - frame)] = (reprojection_error_left, reprojection_error_right)

            # Create the factor
            initialEstimate = gtsam.Values()
            c = gtsam.symbol('c', i)
            fac = gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(link.x_left, link.x_right, link.y), sigma, c, l1,
                                              k_object)

            # insert the initial estimation of the left camera pose
            initialEstimate.insert(c, left_cam_pose)

            # insert the initial estimation of the triangulated point
            initialEstimate.insert(l1, gtsam.Point3(reference_triangulated_point))

            # Extract the error of the factor and store it
            err = fac.error(initialEstimate)
            errors[int(track_last_frame - frame)] = err
            print(f"Error for frame {i} factor: {err}, Reprojection error left: {reprojection_error_left}")
        plot_reprojection_errors(l2_error, errors)

        # for i in range(len(frames)):
        #     initialEstimate = gtsam.Values()
        #     frame = frames[i]
        #     c = gtsam.symbol('c', i)
        #     q = gtsam.symbol('q', i)
        #     link = tracking_db.link(frame, trackId)
        #     stereo_point2d = link.x_left, link.x_right, link.y
        #     if i == len(frames) - 1:
        #         origin = gtsam.Pose3()
        #         graph.add(gtsam.PriorFactorPose3(c, origin, sigma_6d))
        #         # initialEstimate.insert(c, origin)
        #
        #         # graph.add(gtsam.PriorFactorPoint3(q, gtsam.Point3(0, 0, 0)))
        #     fac = gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(link.x_left, link.x_right, link.y), sigma, c, q,
        #                                       k_object)
        #
        #     graph.add(fac)
        #     # print()
        #     calculated_left_camera_matrix = transformations[frame]
        #     rotation = calculated_left_camera_matrix[:3, :3]
        #     translation = calculated_left_camera_matrix[:3, 3]
        #     inverse_rotation = rotation.T
        #     inverse_translation = -inverse_rotation @ translation
        #     left_cam_pose = gtsam.Pose3(gtsam.Rot3(inverse_rotation), gtsam.Point3(inverse_translation))
        #     gtsam_frame = gtsam.StereoCamera(left_cam_pose, k_object)
        #     # triangulate the point in the last frame
        #     triangulated_point = gtsam_frame.backproject(
        #         gtsam.StereoPoint2(stereo_point2d[0], stereo_point2d[1], stereo_point2d[2]))
        #     initialEstimate.insert(c, left_cam_pose)
        #     initialEstimate.insert(q, triangulated_point)
        #     err = fac.error(initialEstimate)
        #     errors[frame] = err

        # optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        # print(fac.error(initialEstimate))
        # print(f"before optimization: {optimizer.error()}")
        # result = optimizer.optimize()
        # print(f"after optimization: {optimizer.error()}")
        # pose_c1 = result.atPose3(gtsam.symbol('c', 0))
        # loc_q1 = result.atPoint3(gtsam.symbol('q', 0))
        # print(f"pose_c1: {pose_c1}")
        # print(f"loc_q1: {loc_q1}")
        # # gtsam.utils.plot_3d_points
        # # error = graph.error(result)
        # print("Error of the graph:", error)

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

    def plot_reprojection_errors(reprojection_errors: Dict[int, Tuple[float, float]], factors_errors):
        frames_keys = sorted(reprojection_errors.keys())
        left_errors = [reprojection_errors[f][0] for f in frames_keys]
        right_errors = [reprojection_errors[f][1] for f in frames_keys]
        factor = [factors_errors[f] for f in frames_keys]

        plt.figure(get_fig_num())
        plt.plot(frames_keys, left_errors, label='Left Camera')
        plt.plot(frames_keys, right_errors, label='Right Camera')
        plt.xlabel('distance from reference frame')
        plt.ylabel('projection Error')
        plt.title('projection Error vs track length')
        plt.legend()
        plt.grid(True)

        plt.figure(get_fig_num())
        plt.plot(frames_keys, factor, label='Factor error')
        plt.xlabel('distance from reference frame')
        plt.ylabel('Factor Error')
        plt.title('Factor Error vs track length')
        plt.legend()
        plt.grid(True)

        plt.figure(get_fig_num())
        plt.plot(left_errors, factor, label='Factor error')
        plt.xlabel('left camera error')
        plt.ylabel('Factor Error')
        plt.title('Factor Error as a function of left camera error')
        plt.legend()
        plt.grid(True)

    trackId = find_random_track_of_length(tracking_db, 12)
    # trackId = 6
    # visualize_track(tracking_db, trackId)
    calculate_plot_reprojection_err(find_random_track_of_length, plot_reprojection_errors, tracking_db, trackId)


def q_5_3(db):
    graph_initial_dict = dict()
    i = 0
    t = read_extrinsic_matrices()
    ts = np.array(t)
    indices = extract_keyframes(db, t)
    # transformations = np.array(read_extrinsic_matrices())
    key_frames = indices[0]
    key_frames = (0, 6)
    # for key_frames in indices:
    # transformations = np.load('transformations_ex3.npy')
    # print(f"keyframes {indices}")
    # print(f"keyframes {indices}")

    graph, initial, cameras_dict, frames_dict = create_factor_graph(key_frames[0], key_frames[1], db)

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()

    print_graph_errors(graph, initial, key_frames, result)
    # Plot the optimized trajectory

    worst_factor = print_biggest_error(graph, initial, result)
    print(f"worst factor: {worst_factor}")

    camera_symbol, location_symbol = get_factor_symbols(worst_factor)

    present_error(camera_symbol, initial, result, location_symbol, worst_factor, title1='Before optimization',
                  title2='After optimization')

    # print(f"Initial estimation projected to camera: {stereo_cam.project(initial_stereo_point)}")

    # cameras
    marginals = gtsam.Marginals(graph, result)
    keys = gtsam.KeyVector()
    # fig = plt.figure(0)
    # axes = fig.add_subplot(projection='3d')

    for cam in cameras_dict.keys():
        keys.append(cameras_dict[cam])
        # print(f"Camera {cam} has marginal covariance: {marginals.marginalCovariance(cameras_dict[cam])}")
    marginals.jointMarginalCovariance(keys).fullMatrix()

    # figs = plt.get_fignums()
    # free_fignum =

    plot_3d_points(fignum=get_fig_num(), values=result, title='3D points')

    plot_trajectory(fignum=get_fig_num(), values=result, marginals=marginals, title='trajectory', scale=1)
    # set_axes_equal(0)




def q_5_4(db):
    bundles = dict()
    key_frames_poses = dict()
    all_transformations = read_extrinsic_matrices()
    cameras_matrix = [M1]
    t = read_extrinsic_matrices()
    keyframes_indices = extract_keyframes(db, t)

    key_frames_poses[0] = gtsam.Pose3()
    for i, key_frames in enumerate(keyframes_indices):
        graph, initial, cameras_dict, frames_dict = create_factor_graph(key_frames[0], key_frames[1], db)
        graph, result = optimize_graph(graph, initial)

        bundles[i] = {'graph': graph, 'initial': initial, 'cameras_dict': cameras_dict, 'frames_dict': frames_dict,
                      'result': result}

        bundle_legnth = key_frames[1] - key_frames[0]
        last_camera_symbol = cameras_dict[key_frames[1]]
        sym = gtsam.DefaultKeyFormatter(last_camera_symbol)

        final_cam_trans = result.atPose3(last_camera_symbol).translation()
        final_cam_rot = result.atPose3(last_camera_symbol).rotation().matrix().T

        # inv = get_inverse(all_transformations[key_frames[0]])
        # rotation = inv[:3, :3]
        # translation = inv[:3, 3]
        # origin = gtsam.Pose3(gtsam.Rot3(rotation), gtsam.Point3(translation))
        #
        # inv = get_inverse(all_transformations[key_frames[1]])
        # rotation = inv[:3, :3]
        # translation = inv[:3, 3]
        # next = gtsam.Pose3(gtsam.Rot3(rotation), gtsam.Point3(translation))
        # relative = origin.between(next)

        # final_cam_trans = relative.translation()
        # final_cam_rot = relative.rotation().matrix().T

        final_cam_trans = -final_cam_rot @ final_cam_trans

        final_matrix = np.hstack((final_cam_rot, final_cam_trans.reshape(-1, 1)))
        last_transformation = cameras_matrix[-1]
        last_transformation = np.vstack((last_transformation, np.array([0, 0, 0, 1])))
        global_transformation = final_matrix @ last_transformation
        global_transformation = global_transformation[:3, :]
        cameras_matrix.append(global_transformation)

    cameras_locations = []
    for cam in cameras_matrix:
        rot = cam[:3, :3]
        t = cam[:3, 3]
        cameras_locations.append(-rot.T @ t)

    cameras_locations2 = []
    for cam in all_transformations:
        rot = cam[:3, :3]
        t = cam[:3, 3]
        cameras_locations2.append(-rot.T @ t)

    distance = []
    for i in range(len(cameras_locations)):
        distance.append(np.linalg.norm(cameras_locations[i] - cameras_locations2[i]))

    print_q_5_4(bundles, cameras_locations, cameras_locations2, distance, keyframes_indices)


if __name__ == '__main__':
    algorithm = 'akaze'

    ORB_PATH = arguments.DATA_HEAD + "docs/ORB/db/db_3359"
    AKAZE_PATH = arguments.DATA_HEAD + "/docs/AKAZE/db/db_3359"
    SIFT_PATH = arguments.DATA_HEAD + "/docs/SIFT/db/db_3359"
    path = r"C:\Users\elyas\University\SLAM video navigation\VAN_ex\code\VAN_ex\dataset\sequences\00"
    if algorithm == 'orb':
        serialized_path = ORB_PATH
    elif algorithm == 'akaze':
        serialized_path = AKAZE_PATH
    elif algorithm == 'sift':
        serialized_path = SIFT_PATH
    else:
        raise ValueError("algorithm should be one of ['orb', 'akaze', 'sift']")

    db = TrackingDB()
    db.load(serialized_path)
    q_5_1(db)
    plt.show()
    q_5_3(db)
    plt.show()
    q_5_4(db)
    plt.show()
