import numpy as np
from scipy.spatial.transform import Rotation as R

def quad_dynamics (t, state, u, params): 
    # Deconstruct the state vector into position, velocity, quaternion, and angular velocity
    position, vel, quat, omega = state[0:3], state[3:6], state[6:10], state[10:13]
    total_thrust, tau = u[0], u[1:4]
 
    #1. Quaternion Kinematics 
    qv, qw = quat[0:3], quat[3]
    d_qv = 0.5 * (qw * omega + np.cross(qv, omega))
    d_qw = -0.5 * np.dot(qv, omega)
    d_quat = np.append(d_qv, d_qw)
 
    #2. Translational Acceleration 
    r = R.from_quat(quat)
    f_body = np.array([0.0, 0.0, total_thrust])
    acc = (r.apply(f_body) / params.m) + np.array([0.0, 0.0, -params.g])
 
    #3. Angular Acceleration 
    alpha = np.linalg.solve(params.Inertia, (tau - np.cross(omega, params.Inertia @ omega)))
 
    return np.concatenate((vel, acc, d_quat, alpha))