from collections import OrderedDict
from jpl_quat_ops import JPLQuaternion, jpl_error_quat
import numpy as np

class StateInfo():
    ''' Stores size and indexes related to the MSCKF state vector '''

    # Represents the indexing of the individual components of the state vector.
    # The slice(a, b) class just represents a python slice [a:b]

    # Attitude, since we are storing the error state the size is 3 rather than the 4 required for the quaternion
    ATT_SLICE = slice(0, 3)
    # Bias gyro
    BG_SLICE = slice(3, 6)
    # Velocity
    VEL_SLICE = slice(6, 9)
    # Bias acc
    BA_SLICE = slice(9, 12)
    # Position
    POS_SLICE = slice(12, 15)

    # Size of the imu state(the above 5 slices)
    IMU_STATE_SIZE = 15

    # Index at which the camera poses start
    CAMERA_POSES_START_INDEX = 15

    # Where within the camera poses the attitude error state is
    CAMERA_ATT_SLICE = slice(0, 3)
    # Where within the camera poses the position error state is
    CAMERA_POS_SLICE = slice(3, 6)

    # Size of the camera pose(3 for attitude, 3 for position)
    CAMERA_STATE_SIZE = 6

class CameraPose():
    ''' Stores camera pose(including attitude and position) in the state vector '''

    def __init__(self, camera_JPLQ_global, global_t_camera, camera_id):
        self.camera_JPLQ_global = camera_JPLQ_global
        self.global_t_camera = global_t_camera
        self.timestamp = 0
        self.camera_id = camera_id

class State():
    '''
    Contains the state vector and the associated covariance for the Kalman Filter

    Within the state vector we keep track of our IMU state(contains attitude, biases, position, etc) and a limited
    amount of Camera Poses. These Poses contains the position and attitude of the camera at some timestamp in the 
    past. They are what allow us to define EKF Update functions linking the past poses to our current pose.

    '''

    def __init__(self):
        
        # Attitude of the IMU. Stores the rotation of the IMU to the global frame as a JPL quaternion
        self.imu_JPLQ_global = JPLQuaternion.identity()

        # Bias Gyro of the IMU
        self.bias_gyro = np.zeros((3, 1), dtype=np.float64)

        # Velocity of the IMU
        self.velocity = np.zeros((3, 1), dtype=np.float64)

        # Bias Acc of the IMU
        self.bias_acc = np.zeros((3, 1), dtype=np.float64)

        # Position of the IMU in the global frame
        self.global_t_imu = np.zeros((3, 1), dtype=np.float64)

        # N camera poses included in the EKF state vector at time-step k
        self.camera_poses = OrderedDict()

        # The covariance matrix of the state of IMU
        self.covariance_ii = np.eye(StateInfo.IMU_STATE_SIZE, dtype=np.float64)

    def add_camera_pose(self, camera_pose):
        self.camera_poses[camera_pose.camera_id] = camera_pose

    def num_camera_poses(self):
        return len(self.camera_poses)

    def calc_camera_pose_index(self, index_within_camera_poses):
        return StateInfo.IMU_STATE_SIZE + index_within_camera_poses * StateInfo.CAMERA_STATE_SIZE

    def get_state_size(self):
        return StateInfo.IMU_STATE_SIZE + self.num_camera_poses() * StateInfo.CAMERA_STATE_SIZE

    def update_state(self, delta_x):
        ''' Update the state vector given a delta_x computed from a measurement update.
        
        Args:
            delta_x: Numpy vector. Has length equal to the error state vector.
        
        '''
        assert(delta_x.shape[0] == self.get_state_size())

        # For every state variable except fot the rotations we can use a simple vector update
        # x' = x + delta_x

        self.bias_gyro += delta_x[StateInfo.BG_SLICE]
        self.velocity += delta_x[StateInfo.VEL_SLICE]
        self.bias_acc += delta_x[StateInfo.BA_SLICE]
        self.global_t_imu += delta_x[StateInfo.POS_SLICE]

        # Attitude requires a special Quaternion update
        # Note because we are using the left jacobians the update needs to be applied from the left side.
        error_quat = JPLQuaternion.from_array(jpl_error_quat(delta_x[StateInfo.ATT_SLICE]))
        self.imu_JPLQ_global = error_quat.multiply(self.imu_JPLQ_global)

        # Now we do same thing for the rest of the state vector(camera poses)
        # enumerate() function tutorial: https://www.runoob.com/python/python-func-enumerate.html
        for idx, camera_pose in enumerate(self.camera_poses.values()):

            delta_x_index = StateInfo.IMU_STATE_SIZE + idx * StateInfo.CAMERA_STATE_SIZE

            # Position update
            pos_start_index = delta_x_index + StateInfo.CAMERA_POS_SLICE.start
            pos_end_index = delta_x_index + StateInfo.CAMERA_POS_SLICE.end
            camera_pose.global_t_camera += delta_x[pos_start_index : pos_end_index]

            # Attitude update
            att_start_index = delta_x_index + StateInfo.CAMERA_ATT_SLICE.start
            att_end_index = delta_x_index + StateInfo.CAMERA_ATT_SLICE.end
            delta_theta = delta_x[att_start_index : att_end_index]
            error_quat = JPLQuaternion.from_array(jpl_error_quat(delta_theta))
            camera_pose.camera_JPLQ_global = error_quat.multiply(camera_pose.camera_JPLQ_global)

    def print_state(self):

        print("Position {}", self.global_t_imu)
