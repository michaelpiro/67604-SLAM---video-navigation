# SLAM Project - Real-Time 3D Mapping Using KITTI Camera Data

## Overview

This project implements a Simultaneous Localization and Mapping (SLAM) system using video footage from two KITTI cameras mounted on a vehicle. The system processes stereo video data to build an accurate, real-time (or near real-time) top-down map of the environment. Our goal is to achieve high-precision mapping while accounting for camera motion and feature matching errors.

We employ a combination of feature detection, geometric estimation, and optimization techniques to achieve accurate localization and mapping in dynamic environments.

## Features

- **Stereo Vision**: Uses two cameras mounted along the Y-axis of the vehicle with a known baseline to calculate depth and generate a 3D map.
  
- **Feature Detection and Matching**: AKAZE descriptors are used to identify key points in images. The system matches points across stereo image pairs and consecutive frames to track the camera’s motion and triangulate the position of 3D points.
  
- **RANSAC for Robust Pose Estimation**: The Random Sample Consensus (RANSAC) algorithm is applied to filter outliers and ensure accurate pose estimation, rejecting invalid matches such as points appearing behind the camera.
  
- **LM Optimization**: We use a Levenberg-Marquardt (LM) optimizer to refine the estimated camera poses, ensuring a smooth and compact pose graph that represents the vehicle’s trajectory over time.
  
- **Loop Closure for Drift Correction**: The system identifies previously visited locations using loop closure detection. When a match is confirmed, the pose graph is recalibrated to reduce drift and improve global accuracy.

## Methodology

1. **Feature Extraction and Matching**:
   - Detect key points in the stereo image pairs using AKAZE descriptors.
   - Match these points across frames and between the left and right cameras.
   - Triangulate 3D points based on stereo correspondences.

2. **Pose Estimation**:
   - Use matched 2D-3D correspondences and the RANSAC algorithm to compute the camera's pose via the Perspective-n-Point (PnP) method.
   - Discard matches that don’t fit the estimated camera motion or lead to unrealistic triangulated points (e.g., behind the camera).

3. **Optimization**:
   - Levenberg-Marquardt (LM) optimization is applied to fine-tune the pose graph, minimizing the overall error and ensuring a compact representation of the poses.

4. **Loop Closure**:
   - Continuously checks for places with high Mahalanobis distance similarity to previously visited locations.
   - Upon detecting a loop closure, recalibrate the pose graph for more accurate localization.

## Future Work

- **Incorporating IMU data**: Integrating inertial measurement data for better pose estimation, especially in areas with few visual features.
- **Advanced Bundle Adjustment**: Implementing full bundle adjustment to further refine the 3D reconstruction.
- **Real-Time Performance Improvements**: Optimizing the system to enhance real-time performance and lower latency.
  
## Resources

For more detailed information on SLAM and the loop closure technique, please refer to this article: [Think Autonomous - Loop Closure in SLAM](https://www.thinkautonomous.ai/blog/loop-closure/).

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/SLAM-Project.git
   ```
2. Follow the installation instructions to set up dependencies (e.g., OpenCV, Eigen, etc.).
3. Run the main program with:
   ```bash
   ./slam_project path/to/kitti/video_data
   ```

if you have any ideas or places for improveents, we would like to hear! you are welcome to contact us at any time , through email:
elyashiv.newman@mail.huji.ac.il
or through github! 
Thanks, Elyashiv and Michael
