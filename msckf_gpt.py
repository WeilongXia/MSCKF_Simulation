# 定义一些基本参数
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R

# Define system parameters
dt = 0.01   # time step (s)
g = np.array([0, 0, -9.81])  # acceleration due to gravity (m/s^2)
sigma_a = 0.05  # accelerometer noise standard deviation (m/s^2)
sigma_w = 0.01  # gyroscope noise standard deviation (rad/s)
sigma_imu = np.array([sigma_a, sigma_a, sigma_a, sigma_w, sigma_w, sigma_w])
sigma_pixel = 1  # pixel noise standard deviation (pixels)
baseline = 0.2  # stereo camera baseline (m)
K = np.array([[200, 0, 320], [0, 200, 240], [0, 0, 1]])  # camera intrinsics matrix


# 定义一些帮助函数
def hat_map(v):
    """Hat mapping for a 3D vector"""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def calculate_residual(x, z, features_cam, camera):
    """Calculate residual between observed and predicted feature locations"""
    p = x[:3]
    v = x[3:6]
    q = R.from_quat(x[6:10])
    p_cam_imu = camera.p_cam_imu
    z_cam = camera.z_cam
    H = np.zeros((2, 9))

    # Iterate over all features
    for i, f in enumerate(features_cam):
        # Calculate feature position in world frame
        f_world = np.hstack((f, np.zeros(1)))
        f_imu = np.dot(p_cam_imu, f_world)
        f_world = np.dot(R.from_quat(q.as_quat())).apply(f_imu - p)

        # Calculate feature position in camera frame
        f_cam = np.dot(inv(camera.T_cam_imu), np.hstack((f_world, 1)))[:3]

        # Calculate feature position in image frame
        f_pixel = np.dot(K, f_cam / f_cam[2])

        # Calculate residual
        r = z[i] - f_pixel[:2]
        H[:, :3] = -np.dot(K, R.from_quat(q.as_quat()).as_matrix())
        H[:, 3:6] = -np.dot(K, hat_map(f_imu - p))
        H[:, 6:9] = -np.dot(K, hat_map(np.dot(R.from_quat(q.as_quat()).as_matrix(), f_cam)))

    return r, H


# 编写主要的蒙特卡洛循环
# Define simulation parameters
N_mc = 100  # number of Monte Carlo trials
N = 5000  # number of time steps

# Monte Carlo loop
for mc in range(N_mc):
    # Generate ground truth trajectory and feature points
    x_true, z_true, features_world = generate_ground_truth(N)

    # Generate measurements
    measurements = generate_measurements(x_true, z_true, features_world)

    # Initialize state and covariance matrix
    x_hat = np.zeros(10)
    P_hat = np.eye(10)

    # Initialize feature tracks
    tracks = []
    for i, f in enumerate(features_world):
        tracks.append(FeatureTrack(i, f))

    # Initialize camera
    camera = Camera(baseline, K)

    # Initialize error history
    errors = []

    # Time loop
    for k in range(N):
        # Propagate state and covariance matrix using IMU measurements
        x_hat, P_hat = propagate_imu(x_hat, P_hat, measurements.imu[k], dt, g)

        # Check if any features have been lost and remove them from the state and covariance matrix
        lost_features = []
        for i, track in enumerate(tracks):
            if not track.is_observed:
                lost_features.append(i)
        x_hat, P_hat = remove_lost_features(x_hat, P_hat, lost_features)

        # Calculate predicted feature locations in camera frame
        features_cam = []
        for track in tracks:
            if track.is_observed:
                f_cam = np.dot(inv(camera.T_cam_imu), np.hstack((track.position, 1)))[:3]
                features_cam.append(f_cam)

        # Calculate residuals and Jacobian for all observed features
        z = measurements.image[k]
        r, H = calculate_residual(x_hat, z, features_cam, camera)

        # Perform measurement update using MSCKF
        x_hat, P_hat, tracks = msckf_update(x_hat, P_hat, r, H, sigma_pixel, tracks)

        # Save error history
        errors.append(np.linalg.norm(x_hat[:3] - x_true[k][:3]))

    # Print simulation results
    print("Monte Carlo trial {0:d} - Final error: {1:.4f} m".format(mc+1, errors[-1]))

# Plot error history
import matplotlib.pyplot as plt
plt.plot(errors)
plt.title("Position error over time")
plt.xlabel("Time step")
plt.ylabel("Position error (m)")
plt.show()
