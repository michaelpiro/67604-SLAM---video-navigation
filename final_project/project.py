####################################################################################################
# Project: SLAM Final Project
# This code will run the system
####################################################################################################


import multiprocessing as mp
import threading
from queue import Queue
import time

# Shared resources
from final_project import arguments
from final_project.analysis import run_analysis
from final_project.backend.GTSam import bundle
from final_project.backend.GTSam.pose_graph import PoseGraph
from final_project.backend.database.tracking_database import TrackingDB
from final_project.arguments import LEN_DATA_SET
from final_project.backend.database.database import run
from final_project.backend.GTSam.bundle import create_single_bundle, optimize_graph

image_database = []
last_keyframe = 0
keyframe_queue = Queue()
bundle_queue = Queue()
pose_graph_queue = Queue()

# Synchronization primitives
database_lock = threading.Lock()
images_processed_event = threading.Event()

IMAGE_POOL_SIZE = 40
TOTAL_IMAGES = LEN_DATA_SET


# def check_for_new_images(db: TrackingDB):
#     # Get the latest image from the database
#     num_images_processed = 0
#     while True:
#         num_images_processed = db.frame_num()
#         with database_lock:
#             global last_keyframe
#             if num_images_processed - last_keyframe >= IMAGE_POOL_SIZE or num_images_processed == TOTAL_IMAGES:
#                 extract_next_keyframes(db)
#         time.sleep(2.0)
#
#
# def tracking_database_creation(image_paths):
#     for image_path in image_paths:
#         # Process the image and update the database
#         processed_image = process_image(image_path)
#         with database_lock:
#             image_database.append(processed_image)
#             if len(image_database) % 100 == 0:
#                 images_processed_event.set()
#         time.sleep(0.1)  # Simulate processing time
#     print("Process 1: Tracking database creation completed.")
#
#
# def keyframes_extraction():
#     batch_start = 0
#     while True:
#         images_processed_event.wait()
#         with database_lock:
#             batch_end = len(image_database)
#             if batch_end - batch_start >= 100:
#                 new_images = image_database[batch_start:batch_start + 100]
#                 batch_start += 100
#             else:
#                 images_processed_event.clear()
#                 continue
#         keyframes = extract_keyframes(new_images)
#         keyframe_queue.put(keyframes)
#         print("Process 2: Extracted keyframes from images {} to {}.".format(batch_start - 100, batch_start))
#         if batch_end == total_images:
#             break
#     print("Process 2: Keyframes extraction completed.")
#
#
# def bundle_optimization():
#     while True:
#         keyframes = keyframe_queue.get()
#         graph, initial_estimate, _ = create_single_bundle(keyframes[0], keyframes[1], db)
#
#         optimized_bundle = optimize_graph(graph, initial_estimate)
#         bundle_queue.put(optimized_bundle)
#         print("Process 3: Bundle optimization completed for one set of keyframes.")
#         if keyframe_queue.empty() and images_processed_event.is_set() == False:
#             break
#     print("Process 3: Bundle optimization completed.")
#
#
# def pose_graph_creation():
#     while True:
#         try:
#             optimized_bundle = bundle_queue.get(timeout=5)
#             pose_graph = create_pose_graph(optimized_bundle)
#             pose_graph_queue.put(pose_graph)
#             print("Process 4: Pose graph created.")
#         except:
#             if images_processed_event.is_set() == False and bundle_queue.empty():
#                 break
#     print("Process 4: Pose graph creation completed.")
#
#
# def loop_closure_optimization():
#     while True:
#         try:
#             pose_graph = pose_graph_queue.get(timeout=5)
#             optimized_pose_graph = optimize_loop_closure(pose_graph)
#             print("Process 5: Loop closure optimization completed.")
#         except:
#             if images_processed_event.is_set() == False and pose_graph_queue.empty():
#                 break
#     print("Process 5: Loop closure optimization completed.")
#
#
# # Placeholder functions for processing
# def process_image(image_path):
#     # Implement your image processing logic here
#     return image_path
#
#
# def extract_keyframes(images):
#     # Implement your keyframe extraction logic here
#     return images
#
#
# def optimize_bundle(keyframes):
#     # Implement your bundle optimization logic here
#     return keyframes
#
#
# def create_pose_graph(optimized_bundle):
#     # Implement your pose graph creation logic here
#     return optimized_bundle
#
#
# def optimize_loop_closure(pose_graph):
#     # Implement your loop closure optimization logic here
#     return pose_graph
#
#
# # Main execution
# if __name__ == "__main__":
#     image_paths = ["image_{}".format(i) for i in range(1, 501)]  # Example image paths
#     total_images = len(image_paths)
#
#     # Start Process 1
#     p1 = threading.Thread(target=tracking_database_creation, args=(image_paths,))
#     p1.start()
#
#     # Start Process 2
#     p2 = threading.Thread(target=keyframes_extraction)
#     p2.start()
#
#     # Start Process 3
#     p3 = threading.Thread(target=bundle_optimization)
#     p3.start()
#
#     # Start Process 4
#     p4 = threading.Thread(target=pose_graph_creation)
#     p4.start()
#
#     # Start Process 5
#     p5 = threading.Thread(target=loop_closure_optimization)
#     p5.start()
#
#     # Wait for all processes to complete
#     p1.join()
#     p2.join()
#     p3.join()
#     p4.join()
#     p5.join()
#
#     print("All processes completed.")

#

# if __name__ == '__main__':
#
#     # load the database
#     serialized_path = arguments.DATA_HEAD + "/docs/SIFT/db/db_3359"
#     # serialized_path = "/Users/mac/67604-SLAM-video-navigation/final_project/SIFT_DB"
#
#     # db = run(serialized_path)
#     db = TrackingDB()
#     db.load(serialized_path)
#
#     key_frames = bundle.get_keyframes(db)
#     pose_graph = PoseGraph()
#     all_bundles = []
#     for key_frame in key_frames:
#         first_frame = key_frame[0]
#         last_frame = key_frame[1]
#         graph, initial, cameras_dict, frames_dict = bundle.create_single_bundle(key_frame[0], key_frame[1], db)
#         graph, result = bundle.optimize_graph(graph, initial)
#
#         bundle_dict = {'graph': graph, 'initial': initial, 'cameras_dict': cameras_dict, 'frames_dict': frames_dict,
#                        'result': result, 'keyframes': key_frame}
#         all_bundles.append(bundle_dict)
#         pose_graph.add_bundle(bundle_dict)
#         print(f"Bundle {key_frame} added to the pose graph")
#
#     pose_graph.optimize()
#     run_analysis(db, all_bundles)



