import pickle
import numpy as np
import gtsam
from gtsam.utils import plot


def calculate_relative_pose_cov(first_frame_symbol, last_frame_symbol, bundle_graph, result):
    """Calculate the relative pose and covariance between two frames."""
    marginals = gtsam.Marginals(bundle_graph, result)
    # Obtain marginal covariances directly
    keys = gtsam.KeyVector()
    keys.append(first_frame_symbol)
    keys.append(last_frame_symbol)

    # calculating the pose
    first_frame_pose = result.atPose3(first_frame_symbol)
    last_frame_pose = result.atPose3(last_frame_symbol)
    relative_pose = first_frame_pose.between(last_frame_pose)

    # Compute the information matrix
    information = marginals.jointMarginalInformation(keys).fullMatrix()
    # Extract the relative covariance
    relative_cov = np.linalg.inv(information[-6:, -6:])
    return marginals, relative_pose, relative_cov


class PoseGraph:
    """Class to the pose and pose graph optimization."""
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.last_pose = None
        self.key_frames = []
        self.graph_scale_factor = 1.0
        self.first_frame_cov = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * self.graph_scale_factor)
        self.result = None

    def add_bundle(self, bundle):
        """Add a new bundle to the pose graph."""

        cameras_dict = bundle['cameras_dict']
        current_graph = bundle['graph']
        current_result = bundle['result']
        start_frame_symbol = cameras_dict[bundle['keyframes'][0]]
        end_frame_symbol = cameras_dict[bundle['keyframes'][1]]
        self.key_frames.append(bundle['keyframes'])

        first_frame_pose = current_result.atPose3(start_frame_symbol)
        last_frame_pose = current_result.atPose3(end_frame_symbol)
        relative_pose = first_frame_pose.between(last_frame_pose)

        try:
            marginals = gtsam.Marginals(current_graph, current_result)

        except Exception as e:
            print(f"Error in calculating the marginals {e}")

        keys = gtsam.KeyVector()
        keys.append(start_frame_symbol)
        keys.append(end_frame_symbol)
        try:
            information = marginals.jointMarginalInformation(keys).fullMatrix()
        except Exception as e:
            print(f"Error in calculating the relative covariance {e}")

        relative_cov = np.linalg.inv(information[-6:, -6:])

        noise_model = gtsam.noiseModel.Gaussian.Covariance(relative_cov)

        if self.last_pose is None:
            # zero_pose = gtsam.Pose3()
            # # rel_pose_zero = zero_pose.between(first_frame_pose)
            # factor = gtsam.PriorFactorPose3(start_frame_symbol, zero_pose, self.first_frame_cov)
            # self.graph.add(factor)
            # self.initial_estimate.insert(start_frame_symbol, first_frame_pose)
            # self.last_pose = first_frame_pose


            zero_pose = gtsam.Pose3()
            # rel_pose_zero = zero_pose.between(first_frame_pose)
            factor = gtsam.PriorFactorPose3(start_frame_symbol, zero_pose, self.first_frame_cov)
            self.graph.add(factor)
            self.initial_estimate.insert(start_frame_symbol, zero_pose)
            self.last_pose = zero_pose

        factor = gtsam.BetweenFactorPose3(start_frame_symbol, end_frame_symbol, relative_pose, noise_model)
        self.graph.add(factor)

        global_pose = self.last_pose.transformPoseFrom(relative_pose)
        self.last_pose = global_pose

        if not self.initial_estimate.exists(start_frame_symbol):
            self.initial_estimate.insert(start_frame_symbol, first_frame_pose)
        if not self.initial_estimate.exists(end_frame_symbol):
            self.initial_estimate.insert(end_frame_symbol, global_pose)

    def optimize(self):
        """Optimize the pose graph."""
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate)
        result = optimizer.optimize()
        self.result = result
        return self.result

    def save(self, filename):
        """Save the pose graph to a file."""
        with open(filename + '.pkl', 'wb') as file:
            pickle.dump(self, file)
        print('PoseGraph saved to', filename)

    @staticmethod
    def load(filename):
        """Load the pose graph from a file."""
        with open(filename, 'rb') as file:
            pose_graph = pickle.load(file)
        print('PoseGraph loaded from', filename)
        return pose_graph

    def plot(self, result):
        """Plot the optimized poses."""
        plot.plot_trajectory(1, self.initial_estimate, title="Initial Poses")
        plot.plot_trajectory(2, result, title="Optimized Poses")
        marg = gtsam.Marginals(self.graph, result)
        plot.plot_trajectory(3, result, marginals=marg, scale=1)

