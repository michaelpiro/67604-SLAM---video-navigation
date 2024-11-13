import pickle

import gtsam
import numpy as np
import cv2

from final_project.algorithms.triangulation import triangulate_last_frame
from final_project.arguments import LEN_DATA_SET
from final_project.backend.database.tracking_database import TrackingDB
from final_project.utils import K, M1, M2, P, Q, rodriguez_to_mat


def load(base_filename):
    """ load TrackingDB to base_filename+'.pkl' file. """
    filename = base_filename + '.pkl'

    with open(filename, 'rb') as file:
        data = pickle.load(file)
        bundles, graphs, results = data
    print('Bundles loaded from', filename)
    return bundles, graphs, results


#
def T_B_from_T_A(T_A, T_B):
    """Calculate the relative transformation between two poses.
    more specifically, how b is seen from a."""
    if T_A.shape == (3, 4):
        T_A = np.vstack((T_A, np.array([0, 0, 0, 1])))
    if T_B.shape == (3, 4):
        T_B = np.vstack((T_B, np.array([0, 0, 0, 1])))

    # return get_inverse(T_A) @ T_B
    return T_B @ get_inverse(T_A)


def get_inverse(T):
    """ Get the inverse of a transformation matrix."""
    inverse_rotation = T[:3, :3].T
    inverse_translation = -inverse_rotation @ T[:3, 3]
    inverse_first = np.hstack((inverse_rotation, inverse_translation.reshape(-1, 1)))
    inverse_first = np.vstack((inverse_first, np.array([0, 0, 0, 1])))
    return inverse_first


def T_to_gtsam_pose(T):
    """Convert a transformation matrix to a shape that fits the gtsam model."""
    return get_inverse(T)


def get_factor_symbols(factor):
    """ Get the symbols of the factor."""
    camera_symbol = factor.keys()[0]
    location_symbol = factor.keys()[1]
    return camera_symbol, location_symbol


def get_factor_point(factor, gt_values):
    """ Get the point of the factor."""
    factor_landmark = get_factor_symbols(factor)[1]
    return gt_values.atPoint3(factor_landmark)


def gtsam_pose_to_T(pose):
    """Convert a gtsam pose to a transformation matrix with shape (3x4)."""
    pose_trans = pose.translation()
    pose_rot = pose.rotation().matrix()

    final_cam_rot = pose_rot.T
    final_cam_trans = -final_cam_rot @ pose_trans

    final_matrix = np.hstack((final_cam_rot, final_cam_trans.reshape(-1, 1)))
    return final_matrix


# TODO: check if this is correct!!! whether joint marginal covariance is the correct function to use
def get_pose_covariance(bundle, values):
    marginals = gtsam.Marginals(bundle, values)
    # keys = gtsam.KeyVector()
    #
    # poses = gtsam.utilities.allPose3s(values)
    # for key in poses.keys():
    #     pose = poses.atPose3(key)
    #     if marginals:
    #         covariance = marginals.marginalCovariance(key)
    #     else:
    #         covariance = None
    # marginals.jointMarginalCovariance(keys).fullMatrix()



def calculate_relative_transformation(db: TrackingDB, first_frame_idx, last_frame_idx):
    """Calculate the realtive transformations between the first and every frame between
        the first and the last frame."""
    transformations = dict()
    transformations[0] = np.vstack((M1, np.array([0, 0, 0, 1])))
    # diff_coeff = np.zeros((5, 1))
    for i in range(1, last_frame_idx - first_frame_idx + 1):
        T = calc_rel_T(db, i)
        if T is None:
            raise Exception("PnP failed")
        new_trans = get_inverse(T)
        transformations[i] = get_inverse(new_trans @ transformations[i - 1])
    return transformations


def calc_rel_T(db: TrackingDB, frame):
    """Calculate the realtive transformations between 2 consecutive frames."""

    if frame == 0:
        return M1

    diff_coeff = np.zeros((5, 1))
    current_frame_id = frame
    prev_frame_id = current_frame_id - 1

    # get the relevant tracks
    common_tracks = list(set(db.tracks(current_frame_id)).intersection(
        set(db.tracks(prev_frame_id))))
    links_prev_frame = [db.link(prev_frame_id, track_id) for track_id in common_tracks]
    links_current_frame = [db.link(current_frame_id, track_id) for track_id in common_tracks]

    # calculate the transformation between the two frames
    # traingulate the links
    triangulated_links = triangulate_last_frame(db, P, Q, links_prev_frame)
    if len(triangulated_links) < 4:
        raise Exception("Not enough points to triangulate and perform PnP")
    # calculate the transformation

    links_current_frame = np.array(links_current_frame)
    img_points = np.array([(link.x_left, link.y) for link in links_current_frame])
    # links_current_frame = np.array(links_current_frame)
    success, rotation_vector, translation_vector = cv2.solvePnP(triangulated_links, img_points, K,
                                                                distCoeffs=diff_coeff, flags=cv2.SOLVEPNP_EPNP)

    T = rodriguez_to_mat(rotation_vector, translation_vector) if success else None
    if T is None:
        raise Exception("PnP failed")
    return T


def calculate_global_transformation(db: TrackingDB, first_frame_idx, last_frame_idx):
    """Calculate the global transformation between the first and the last frame."""
    # transformations = calculate_relative_transformation(db, first_frame_idx, last_frame_idx)
    rel_T = []
    for i in range(LEN_DATA_SET):
        T = calc_rel_T(db, i)
        if i == 0:
            rel_T.append(T)
        else:
            rel_T.append(T @ np.vstack((rel_T[-1], np.array([0, 0, 0, 1]))))
    return rel_T

def calculate_all_pnp_rel_transformation(db: TrackingDB):
    """Calculate the global transformation between the first and the last frame."""
    # transformations = calculate_relative_transformation(db, first_frame_idx, last_frame_idx)
    rel_T = []
    for i in range(LEN_DATA_SET):
        rel_T.append(calc_rel_T(db, i))
    return rel_T


def get_index_list(result):
    """
    Extract and sort camera frame indices from the result.

    :param result: Current optimized pose estimates.
    :return: Sorted list of camera frame indices.
    """
    index_list = []
    for key in result.keys():
        cam_number = int(gtsam.DefaultKeyFormatter(key)[1:])
        index_list.append(cam_number)
    index_list.sort()
    return index_list


def get_camera_location_from_gtsam(pose: gtsam.Pose3):
    """
    Extract the translation component (location) from a GTSAM Pose3 object.

    :param pose: GTSAM Pose3 object.
    :return: Translation vector representing the camera location.
    """
    return pose.translation()


def get_locations_from_gtsam(result):
    """
    Extract camera locations from the GTSAM result.

    :param result: Current optimized pose estimates.
    :return: List of camera locations.
    """
    locations = []
    index_list = get_index_list(result)
    for index in index_list:
        pose = result.atPose3(gtsam.symbol('c', index))
        location = get_camera_location_from_gtsam(pose)
        # Verify that the location matches the pose's translation
        assert location[0] == pose.x()
        assert location[1] == pose.y()
        assert location[2] == pose.z()
        locations.append(location)
    return locations


def get_poses_from_gtsam(result):
    """
    Extract camera locations from the GTSAM result.

    :param result: Current optimized pose estimates.
    :return: List of camera locations.
    """
    rotations = dict()
    index_list = get_index_list(result)
    for i,index in enumerate(index_list):
        pose = result.atPose3(gtsam.symbol('c', index))
        rotation = pose.rotation().matrix().T
        # Verify that the location matches the pose's translation
        rotations[i] = rotation
    rotation_list = [rotations[i] for i in range(len(rotations))]
    return rotation_list


def calculate_dist_traveled(transformations):
    """Calculate the distance traveled between the first and the last frame."""
    dist = 0
    accumulate_distance = [0]
    for i in range(1, len(transformations)):
        T_B = transformations[i]
        T_A = transformations[i - 1]
        rel_t = T_B_from_T_A(T_A, T_B)
        R = rel_t[:, :3]
        t = rel_t[:, 3:]
        loc = (-R.transpose() @ t).reshape((3))
        dist = np.linalg.norm(loc)
        accumulate_distance.append(accumulate_distance[-1] + dist)
    return accumulate_distance


def get_bundle_global_mat(result, global_transformation):
    """
    Extract camera locations from the GTSAM result.

    :param result: Current optimized pose estimates.
    :return: List of camera locations.
    """
    index_list = get_index_list(result)
    bundle_global_mat = []
    for index in index_list:
        pose = result.atPose3(gtsam.symbol('c', index))
        location = get_camera_location_from_gtsam(pose)
        rotation = pose.rotation().matrix().T

def get_symbol(index):
    """
    Generate a GTSAM symbol for a given camera index.

    :param index: Camera frame index.
    :return: GTSAM symbol.
    """
    return gtsam.symbol('c', index)

def save(data, base_filename):
    """ save TrackingDB to base_filename+'.pkl' file. """
    if data is not None:
        filename = base_filename + '.pkl'
        with open(filename, "wb") as file:
            pickle.dump(data, file)
        print('bundle saved to: ', filename)
