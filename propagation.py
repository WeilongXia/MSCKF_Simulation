from sys import ps1
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

    Use 4th order Runge-Kutta numerical integration for position and velocity to get more accuracy.

    Note that the biases keep constant after last update step
    '''

    # reference: https://zhuanlan.zhihu.com/p/107032156

    dt = imu_measurement.time_interval
    unbiased_gyro = imu_measurement.angular_vel - self.state.bias_gyro
    unbiased_acc = imu_measurement.linear_acc - self.state.bias_acc

    gyro_norm = np.linalg.norm(unbiased_gyro)

    p0 = self.state.global_t_imu
    q0 = self.state.imu_JPLQ_global
    v0 = self.state.velocity

    omega = jpl_omega(unbiased_gyro)
    
    # Using zeroth order quaternion integrator from 
    # (http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf) Eq. 101, 103
    if(gyro_norm > 1e-5):
        q1 = (np.cos(gyro_norm * dt * 0.5) * np.eye(4) +
              1 / gyro_norm * np.sin(gyro_norm * dt * 0.5) * omega) @ q0.q
        q1_2 = (np.cos(gyro_norm * dt * 0.25) * np.eye(4) +
              1 / gyro_norm * np.sin(gyro_norm * dt * 0.25) * omega) @ q0.q
    else:
        q1 = (np.eye(4) + 0.5 * dt * omega) @ q0.q
        q1_2 = (np.eye(4) + 0.25 * dt * omega) @ q0.q

    R1 = JPLQuaternion.from_array(q1).rotation_matrix().T
    R1_2 = JPLQuaternion.from_array(q1_2).rotation_matrix().T

    k1_v_dot = q0.rotation_matrix().T @ unbiased_acc + self.gravity
    k1_p_dot = v0

    # k2 = f(tn + dt / 2, yn + k1 * dt / 2)
    k1_v = v0 + k1_v_dot * dt / 2
    k2_v_dot = R1_2 @ unbiased_acc + self.gravity

    k2_p_dot = k1_v

    # k3 = f(tn + dt / 2, yn + k2 * dt / 2)
    k2_v = v0 + k2_v_dot * dt / 2
    k3_v_dot = R1_2 @ unbiased_acc + self.gravity

    k3_p_dot = k2_v

    # k4 = f(tn + dt, yn + k3 * dt)
    k3_v = v0 + k3_v_dot * dt
    k4_v_dot = R1 @ unbiased_acc + self.gravity

    k4_p_dot = k3_v

    # y(n+1) = y(n) + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    v1 = v0 + dt / 6 * (k1_v_dot + 2 * k2_v_dot + 2 * k3_v_dot + k4_v_dot)
    p1 = p0 + dt / 6 * (k1_p_dot + 2 * k2_p_dot + 2 * k3_p_dot + k4_p_dot)

    self.state.imu_JPLQ_global = JPLQuaternion.from_array(q1)
    self.state.global_t_imu = p1
    self.state.velocity = v1


def propagate(self, imu_buffer):
    '''
    EKF covariance matrix is propagated.
    '''

