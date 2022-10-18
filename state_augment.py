from jpl_quat_ops import JPLQuaternion
from math_utilities import skew_matrix, symmeterize_matrix
from state import CameraPose, State
from state import StateInfo
import numpy as np

def augment_camera_pose(self):
    imu_R_global = self.state.imu_JPLQ_global.rotation_matrix()
    camera_R_imu = self.camera_calib.imu_R_camera.T
    imu_t_camera = self.camera_calib.imu_t_camera

    # Compute the pose(position and orientation) of the camera in the global frame
    # A.I. Mourikis, S.I. Roumeliotis: "A Multi-state Constraint Kalman Filter for
    # Vision-Aided Inertial Navigation," Eq. 14
    camera_R_global = camera_R_imu @ imu_R_global

    global_t_imu = self.state.global_t_imu
    global_R_imu = imu_R_global.T
    global_t_camera = global_t_imu + global_R_imu @ imu_t_camera

    camera_JPLQ_global = JPLQuaternion.from_rot_mat(camera_R_global)

    cur_state_size = self.state.get_state_size()

    # This jacobian stores the partial derivatives of the camera pose(6 states, 
    # including 3 for position and 3 for orientation) with respect to the current
    # state vector
    jacobian = np.zeros((StateInfo.CAMERA_STATE_SIZE, cur_state_size), dtype=np.float64)
    # Note that it's almost all zeros except for with respect to the current pose(quaternion and position)

    # A.I. Mourikis, S.I. Roumeliotis: "A Multi-state Constraint Kalman Filter for
    # Vision-Aided Inertial Navigation," Eq. 16
    jacobian[StateInfo.CAMERA_ATT_SLICE, StateInfo.ATT_SLICE] = camera_R_imu
    jacobian[StateInfo.CAMERA_POS_SLICE, StateInfo.ATT_SLICE] = skew_matrix(global_R_imu @ imu_t_camera)
    jacobian[StateInfo.CAMERA_POS_SLICE, StateInfo.POS_SLICE] = np.eye(3)

    new_state_size = cur_state_size + StateInfo.CAMERA_STATE_SIZE

    # A.I. Mourikis, S.I. Roumeliotis: "A Multi-state Constraint Kalman Filter for
    # Vision-Aided Inertial Navigation," Eq. 15
    augmented_matrix = np.eye(new_state_size, cur_state_size)
    augmented_matrix[cur_state_size:, :] = jacobian

    new_covariance = augmented_matrix @ self.state.covariance @ augmented_matrix.T
    # Helps with numerical problems
    new_cov_sym = symmeterize_matrix(new_covariance)

    # Add the camera pose and set the new covariance matrix which includes it
    self.camera_id += 1
    camera_pose = CameraPose(camera_JPLQ_global, global_t_camera, self.camera_id)
    self.state.add_camera_pose(camera_pose)
    self.state.covariance = new_cov_sym

    return self.camera_id

    