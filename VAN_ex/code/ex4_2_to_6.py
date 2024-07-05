from VAN_ex.code import ex3
from tracking_database import TrackingDB
import numpy as np
import cv2
import pickle
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import os
import ex2

LEN_DATA_SET = len(os.listdir(DATA_PATH + 'image_0'))


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

    def find_track_of_minimal_length(tracking_db: TrackingDB, length: int) -> Optional[int]:
        for trackId, frames in tracking_db.trackId_to_frames.items():
            if len(frames) >= length:
                return trackId
        return None

    def visualize_track(tracking_db: TrackingDB, trackId: int):
        frames = tracking_db.frames(trackId)
        for i in range(0, 6, 1):
            frameId = frames[i]
            img, _ = ex2.raed_image(frameId)
            x_left, y = get_feature_location(tracking_db, frameId, trackId)
            x_min = int(max(x_left - 10, 0))
            x_max = int(min(x_left + 10, img.shape[1]))
            y_min = int(max(y - 10, 0))
            y_max = int(min(y + 10, img.shape[0]))
            cutout = img[y_min:y_max, x_min:x_max]

            plt.subplot(6, 2, i)
            plt.imshow(img, cmap='gray')
            plt.scatter(x_left, y, color='red')  # Center of the cutout

            plt.subplot(6, 2, i + 1)
            plt.imshow(cutout, cmap='gray')
            plt.scatter([10], [10], color='red')  # Center of the cutout
            plt.title(f"Frame {frameId}, Track {trackId}")
        plt.show()


    minimal_length = 6
    trackId = find_track_of_minimal_length(tracking_db, minimal_length)
    if trackId is None:
        print(f"No track of length {minimal_length} found")
    else:

        print(f"Track of length {minimal_length} found: {trackId}")


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
        plt.plot(frames, counts, marker='o')
        plt.xlabel('Frame ID')
        plt.ylabel('Number of Outgoing Tracks')
        plt.title('Connectivity Graph: Outgoing Tracks per Frame')
        plt.grid(True)
        plt.show()

    # Compute outgoing tracks
    outgoing_tracks = compute_outgoing_tracks(tracking_db)

    # Plot the connectivity graph
    plot_connectivity_graph(outgoing_tracks)


def q_4_5(tracking_db: TrackingDB):
    def plot_inliers_percentage_graph(inliers_percentage_dict: Dict[int, float]):
        frames = sorted(inliers_percentage_dict.keys())
        percentages = [inliers_percentage_dict[frame] for frame in frames]

        plt.figure(figsize=(10, 6))
        plt.plot(frames, percentages, marker='o')
        plt.xlabel('Frame ID')
        plt.ylabel('Percentage of Inliers')
        plt.title('Percentage of Inliers per Frame')
        plt.grid(True)
        plt.show()

    inliers_percentage = {}
    for frame_idx in range(LEN_DATA):
        img_l, img_r = ex2.read_images(frame_idx)
        kp0, kp1, desc0, desc1, matches = ex3.extract_kps_descs_matches(img_l, img_r)
        inliers, outliers = ex3.extract_inliers_outliers(kp0, kp1, matches)
        inliers_percentage[frame_idx] = (len(inliers) / (len(inliers) + len(outliers))) * 100
    # Compute inliers percentage
    # inliers_percentage = compute_inliers_percentage(tracking_db)

    # Plot the inliers percentage graph
    plot_inliers_percentage_graph(inliers_percentage)


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
        plt.show()

    # Calculate track lengths
    track_lengths = calculate_track_lengths(tracking_db)

    # Plot the track length histogram
    plot_track_length_histogram(track_lengths)
