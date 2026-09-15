import numpy as np
from scipy.spatial.transform import Rotation as R

def position_controller(pos, vel, target_pos, params, Kp_pos, Kd_pos, yaw_des=0.0):
    pos_error = target_pos - pos
    vel_error = -vel   # target velocity assumed zero (station-keeping)
 
    desired_accel = Kp_pos * pos_error + Kd_pos * vel_error
 
    # Specific force the thrust vector must supply: cancel gravity, then add
    # whatever extra acceleration is needed to close the position error.
    f_des = desired_accel + np.array([0.0, 0.0, params.g])
 
    thrust_cmd = params.m * np.linalg.norm(f_des)
    if thrust_cmd < 1e-6:
        return params.m * params.g, np.array([0.0, 0.0, yaw_des])
 
    z_b_des = f_des / np.linalg.norm(f_des)
 
    # Desired heading direction projected into the horizontal plane, used
    # only to resolve yaw about the tilted thrust axis.
    x_c_des = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
    y_b_des = np.cross(z_b_des, x_c_des)
    norm_y = np.linalg.norm(y_b_des)
    if norm_y < 1e-6:
        # z_b_des is nearly vertical-aligned with x_c_des (extreme tilt,
        # >~90 deg) -- not expected for gentle setpoints. Fall back to a
        # safe default axis to avoid a divide-by-zero.
        y_b_des = np.array([0.0, 1.0, 0.0])
    else:
        y_b_des = y_b_des / norm_y
    x_b_des = np.cross(y_b_des, z_b_des)
 
    R_des = np.column_stack((x_b_des, y_b_des, z_b_des))
    desired_euler = R.from_matrix(R_des).as_euler('xyz')
 
    return thrust_cmd, desired_euler