import numpy as np

def camera_measurement(self, true_pose, points_w, fov):
    '''
    Implementation of feature points being observed by camera.

    Args:
        true_pose: the true pose of camera(including position and rotation)
        points_w: points in world frame, which is generated by gen_map module
        fov: the field of view of our camera

    Returns:
        points which are in camera's FOV
    '''