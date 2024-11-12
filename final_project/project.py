####################################################################################################
# Project: SLAM Final Project
# This code will run the system
####################################################################################################


# Shared resources
import copy

from final_project import arguments
from final_project.analysis import run_analysis, plot_loc_deg_uncertainties, calculate_camera_locations
from final_project.backend.GTSam import bundle
from final_project.backend.GTSam.pose_graph import PoseGraph
from final_project.backend.database.tracking_database import TrackingDB
from final_project.backend.database.database import *
from final_project.backend.loop.loop_closure import *
from final_project.utils import *

PG_LC_PKL = "pg_lc"
PG_BEFORE_L_C_PKL = "pg_before_l_c"
ALL_BUNDLES_PKL = "all_bundles.pkl"
DEFAULT_DB_NAME = "new_db"

def run_project(path_to_db = None, path_to_bundles = None, path_to_pg = None, path_to_pg_lc = None):
    # load or create the database
    if path_to_db:
        db = TrackingDB()
        db.load(path_to_db)
    else:
        db = run(DEFAULT_DB_NAME)

    # load or create the bundles
    if path_to_bundles:
        all_bundles = load(path_to_bundles)
    else:
        all_bundles = []
        # get keyframes to create bundles
        key_frames = bundle.get_keyframes(db)

        # create all the bundles
        for key_frame in key_frames:

            # create a single bundle
            graph, initial, cameras_dict, frames_dict = bundle.create_single_bundle(key_frame[0], key_frame[1], db)

            # optimize the factor graph
            graph, result = bundle.optimize_graph(graph, initial)

            # create a dictionary to store the bundle
            bundle_dict = {'graph': graph, 'initial': initial, 'cameras_dict': cameras_dict, 'frames_dict': frames_dict,
                           'result': result, 'keyframes': key_frame}

            # add the bundle to the pose graph
            all_bundles.append(bundle_dict)

        # print(f"Bundle {key_frame} added to the pose graph")

        # save the bundles
        save(all_bundles, ALL_BUNDLES_PKL)

    # load or create PoseGraph object
    if path_to_pg_lc:
        pg_lc = load(path_to_pg)
    else:
        # create PoseGraph object
        pg_lc = PoseGraph()

        # add all the bundles to the pose graph
        for bundle_dict in all_bundles:
            pg_lc.add_bundle(bundle_dict)

        # optimize the pose graph
        pg_lc.optimize()

        # save the pose graph
        pg_lc.save(PG_BEFORE_L_C_PKL)

        # perform loop closure
        pg_lc, factor_pose_graph, result = find_loops(pg_lc, db)

        # save the pose graph with loop closure
        pg_lc.save(PG_LC_PKL)


    # load the pose graph before loop closure
    if path_to_pg:
        pg = load(path_to_pg)
    else:
        pg = load(PG_BEFORE_L_C_PKL)

    # run the analysis
    run_analysis(db, all_bundles, pg_lc, pg)




if __name__ == '__main__':
    # path_to_db = "SIFT_DB"
    path_to_db = None
    path_to_bundles = None
    path_to_pg = None
    path_to_pg_lc = None

    run_project(path_to_db, path_to_bundles, path_to_pg, path_to_pg_lc)





