from final_project.backend.database.tracking_database import TrackingDB
import numpy as np


def linear_least_squares_triangulation(P, Q, kp_left, kp_right):
    """
    Linear least squares triangulation
    :param P: 2D point in image 1
    :param Q: Camera matrix of image 2
    :param kp_left: key points in image 1
    :param kp_right: key points in image 2
    :return: 3D point
    """
    A = np.zeros((4, 4))
    p_x, p_y = kp_left
    q_x, q_y = kp_right
    A[0] = P[2] * p_x - P[0]
    A[1] = P[2] * p_y - P[1]
    A[2] = Q[2] * q_x - Q[0]
    A[3] = Q[2] * q_y - Q[1]
    _, _, V = np.linalg.svd(A)
    if V[-1, 3] == 0:
        return V[-1, :3] / (V[-1, 3] + 1e-20)
    return V[-1, :3] / V[-1, 3]


def triangulate_last_frame(tracking_db: TrackingDB, p, q, links=None):
    """
    Triangulate the matched points using OpenCV
    """
    if links is None:
        links = tracking_db.all_last_frame_links()
    x = np.zeros((len(links), 3))
    for i in range(len(links)):
        x_left, x_right, y = links[i].x_left, links[i].x_right, links[i].y
        p_left, p_right = (x_left, y), (x_right, y)
        x[i] = linear_least_squares_triangulation(p, q, p_left, p_right)
    return x


def triangulate_links(links, p, q):
    """
    Triangulate the matched points using OpenCV
    """
    x = np.zeros((len(links), 3))
    for i, link in enumerate(links):
        left_point = link.x_left, link.y
        right_point = link.x_right, link.y
        x[i] = linear_least_squares_triangulation(p, q, left_point, right_point).T
    return x
