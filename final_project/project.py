####################################################################################################
# Project: SLAM Final Project
# This code will run the system
####################################################################################################


# Shared resources
import copy

from final_project import arguments
from final_project.analysis import run_analysis
from final_project.backend.GTSam import bundle
from final_project.backend.GTSam.pose_graph import PoseGraph
from final_project.backend.database.tracking_database import TrackingDB
from final_project.backend.database.database import *
from final_project.backend.loop.loop_closure import *




if __name__ == '__main__':
    # load the database
    serialized_path = SIFT_DB_PATH

    # db = run(serialized_path)
    db = TrackingDB()
    db.load(serialized_path)

    key_frames = bundle.get_keyframes(db)
    pose_graph = PoseGraph()
    # all_bundles = []
    # for key_frame in key_frames:
    #     first_frame = key_frame[0]
    #     last_frame = key_frame[1]
    #     graph, initial, cameras_dict, frames_dict = bundle.create_single_bundle(key_frame[0], key_frame[1], db)
    #     graph, result = bundle.optimize_graph(graph, initial)
    #
    #     bundle_dict = {'graph': graph, 'initial': initial, 'cameras_dict': cameras_dict, 'frames_dict': frames_dict,
    #                    'result': result, 'keyframes': key_frame}
    #     all_bundles.append(bundle_dict)
    #     pose_graph.add_bundle(bundle_dict)

        # print(f"Bundle {key_frame} added to the pose graph")

    save(all_bundles,"sift_bundles")
    save(pose_graph,"sift_pose_graph")
    pose_graph.optimize()
    pose_graph_no_lc = PoseGraph()
    pose_graph_no_lc.graph = copy.deepcopy(pose_graph.graph)
    pose_graph_no_lc.initial_estimate = copy.deepcopy(pose_graph.initial_estimate)
    pose_graph_no_lc.result = copy.deepcopy(pose_graph.result)

    find_loops(pose_graph, db)

    run_analysis(db, all_bundles, pose_graph,pose_graph_no_lc)



