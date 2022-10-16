from turtle import st
from math_utilities import skew_matrix
import numpy as np
from jpl_quat_ops import JPLQuaternion, jpl_omega
from state import State
from state import StateInfo

def compute_F(self, imu):
    ''' 
    Computes the transition matrix(F) for the extended Kalman Filter

    Args:
        unbiased_gyro_meas
        unbiased_acc_measurement

    Returns:
        The transition matrix contains the jacobians of our process model with respect to our current state.

        The jacobians fot this function can be found in A.I. Mourikis, S.I. Roumeliotis: "A Multi-state Constraint
        Kalman Filter for Vision-Aided Inertial Navigation"
    '''
    unbiased_gyro_meas = imu.angular_vel - self.state.bias_gyro
    unbiased_acc_meas = imu.linear_acc - self.state.bias_acc

    state_info = StateInfo
    F = np.zeros((state_info.IMU_STATE_SIZE, state_info.IMU_STATE_SIZE), dtype=np.float64)

    imu_SO3_global = self.state.imu_JPLQ_global.rotation_matrix()

    # This matrix can be found in A.I. Mourikis, S.I. Roumeliotis: "A Multi-state Constraint Kalman Filter for
    # Vision-Aided Inertial Navigation,"
    # How to read the indexing. The first index shows the value we are taking the partial derivative of and the
    # second index is which partial derivative
    # E.g. F[state_info.ATT_SLICE, state_info.BG_SLICE] is the partial derivative of the attitude with respect
    # to the gyro bias.
    # And note that wo do not take account in the effects of the planets's rotation
    F[state_info.ATT_SLICE, state_info.ATT_SLICE] = -skew_matrix(unbiased_gyro_meas)
    F[state_info.ATT_SLICE, state_info.BG_SLICE] = -np.eye(3)
    F[state_info.VEL_SLICE, state_info.ATT_SLICE] = -imu_SO3_global.transpose() @ skew_matrix(unbiased_acc_meas)
    F[state_info.VEL_SLICE, state_info.BA_SLICE] = -imu_SO3_global.transpose()
    F[state_info.POS_SLICE, state_info.VEL_SLICE] = np.eye(3)

    return F


def compute_G(self):
    '''
    Computes the transition matrix(G) with respect to the control input

    Returns:
        A numpy matrix(15x12) containing the jacobians with respect to the IMU measurement noise and random walk noise
    '''

    G = np.zeros((15, 12), dtype=np.float64)
    imu_SO3_global = self.state.imu_JPLQ_global.rotation_matrix()

    state_info = StateInfo
    # order of noise: gyro_m, gyro_b, acc_m, acc_b
    G[state_info.ATT_SLICE, 0:3] = -np.eye(3)
    G[state_info.BG_SLICE, 3:6] = np.eye(3)
    G[state_info.VEL_SLICE, 6:9] = -imu_SO3_global.transpose()
    G[state_info.BA_SLICE, 9:12] = np.eye(3)

    return G

def integrate(self, imu_measurement):
    '''
    Integrate IMU measurement to obtain nominal state of the system.

    Use 5th order Runge-Kutta numerical integration to get more accuracy.

    Note that the biases keep constant after last update step
    '''

    dt = imu_measurement.time_interval
    unbiased_gyro = imu_measurement.angular_vel - self.state.bias_gyro
    unbiased_acc = imu_measurement.linear_acc - self.state.bias_acc

    gyro_norm = np.linalg.norm(unbiased_gyro)

    p0 = self.state.global_t_imu
    q0 = self.state.imu_JPLQ_global
    v0 = self.state.velocity

    omega = jpl_omega(unbiased_gyro)
    # todo


def propagate(self, imu_buffer):
    '''
    EKF covariance matrix is propagated.
    '''

