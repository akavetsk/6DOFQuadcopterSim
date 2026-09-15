import numpy as np 
from parameters import QuadParams

# Using X-Configuratiion for quadcopter motoring mixing 
# Define the motor positions 
motor_positions = np.array([
    [ 1.0,  1.0],
    [-1.0,  1.0],
    [-1.0, -1.0],
    [ 1.0, -1.0],
]) / np.sqrt(2.0)

# Determine spin directions for each motor (1 for CCW, -1 for CW) Adajacent motors are spinning in opposite directions to cancel out torque.
motor_directions = np.array([1, -1, 1, -1])

def mixing_matrix(params):
    """
    Computes the motor mixing matrix for a quadcopter in X-configuration.
    
    Args:
        params: An instance of QuadParams containing quadcopter parameters.
        
    Returns:
        A 4x4 numpy array representing the mixing matrix.
    """
    l = params.l
    kf = params.kf
    km = params.km

    x = motor_positions[:, 0] * l
    y = motor_positions[:, 1] * l
 
    M = np.array([
        np.ones(4),
        y,
        -x,
        motor_directions * (km / kf),
    ])
    return M

def motor_speeds_from_wrench(thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd, params):
    """
    Inverse mixing (controller -> motors).
 
    Given a desired total thrust (N) and roll/pitch/yaw torques (N*m),
    solve for the 4 individual motor thrusts, then back out motor speeds
    from f = kf * omega^2.
 
    Returns
    -------
    f     : (4,) individual motor thrusts (N), clipped to >= 0
    omega : (4,) individual motor speeds (rad/s)
    """
    M = mixing_matrix(params)
    wrench = np.array([thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd])
    f = np.linalg.solve(M, wrench)
 
    if np.any(f < 0):
        # A motor can't produce negative thrust. This means the requested
        # wrench is infeasible for this vehicle (too much torque relative
        # to available thrust). Clipping here is a simple saturation model;
        # real allocation would prioritize attitude over throttle.
        f = np.clip(f, 0.0, None)
 
    omega = np.sqrt(f / params.kf)
    return f, omega
 
 
def wrench_from_motor_speeds(omega, params):
    """
    Forward mixing (motors -> airframe).
 
    Given the 4 actual motor speeds (rad/s), compute the thrust and torque
    actually delivered to the airframe. This is what should be fed into
    `u` for quad_dynamics -- it reflects reality (including any saturation
    that happened upstream), not just the original command.
 
    Returns
    -------
    u : (4,) [total_thrust, tau_roll, tau_pitch, tau_yaw] -- ready to pass to quad_dynamics
    f : (4,) individual motor thrusts (N)
    """
    f = params.kf * omega**2
    M = mixing_matrix(params)
    u = M @ f
    return u, f
