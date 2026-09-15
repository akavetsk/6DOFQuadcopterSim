import numpy as np
from scipy.spatial.transform import Rotation as R
 
"""
Quaternion-based PD attitude controller.
 
Control law:
    tau = -Kp * att_error_vec  -  Kd * omega
  
This is a regulation controller (drives to a fixed desired attitude with
zero desired body rate) -- no feedforward for a moving setpoint yet.
"""
 
 
def euler_to_quat(roll, pitch, yaw):
    """Convert roll/pitch/yaw (rad) to a quaternion [qx, qy, qz, qw]."""
    return R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
 
 
def attitude_pd_controller(quat, omega, desired_euler, Kp, Kd):
    """
    Parameters
    ----------
    quat          : (4,) current attitude quaternion [qx, qy, qz, qw]
    omega         : (3,) current body angular velocity [wx, wy, wz] (rad/s)
    desired_euler : (3,) desired [roll, pitch, yaw] (rad)
    Kp, Kd        : (3,) proportional / derivative gains for [roll, pitch, yaw] axes
 
    Returns
    -------
    tau : (3,) commanded torque [tau_x, tau_y, tau_z] (N*m) -- this maps
          directly to [roll_cmd, pitch_cmd, yaw_cmd] in compute_command.
    """
    q_desired = euler_to_quat(*desired_euler)
 
    r_current = R.from_quat(quat)
    r_desired = R.from_quat(q_desired)
 
    r_error = r_desired.inv() * r_current
    q_error = r_error.as_quat()          # [x, y, z, w]
 
    if q_error[3] < 0:
        # Ensure we rotate the short way around (scalar part positive)
        q_error = -q_error
 
    att_error_vec = q_error[0:3]
 
    tau = -Kp * att_error_vec - Kd * np.asarray(omega)
    return tau