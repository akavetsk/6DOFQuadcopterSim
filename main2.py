import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R

from parameters import QuadParams
from dynamics import quad_dynamics
from motormixing import motor_speeds_from_wrench, wrench_from_motor_speeds
from attitude_controller import attitude_pd_controller, euler_to_quat
from position_controller import position_controller

# Call Parameters
params = QuadParams()

# --- Attitude controller setup (inner loop) ---
# Gains per axis [roll, pitch, yaw]. Roll/pitch share inertia (0.015),
# yaw has higher inertia (0.028) and typically wants softer gains.
#
# Cascaded control requires the inner (attitude) loop to be meaningfully
# faster than the outer (position) loop -- otherwise the outer loop's
# assumption that attitude tracks "instantly" breaks down and the two
# loops couple into an oscillation. Tuned here for ~5x separation from
# the position loop's natural frequency (see position gains below).
Kp_att = np.array([1.5, 1.5, 1.0])
Kd_att = np.array([0.35, 0.35, 0.30])

# --- Position controller setup (outer loop) ---
Kp_pos = np.array([2.0, 2.0, 4.0])   # [x, y, z]
Kd_pos = np.array([2.5, 2.5, 3.0])
target_pos = np.array([0, 0, 5.0])   # move diagonally and climb to 5 m
yaw_des = 0.0

# Define the time span and initial conditions -- start level, at the origin
pos0 = np.array([0.0, 0.0, 0.0])
vel0 = np.array([0.0, 0.0, 0.0])
quat0 = euler_to_quat(0.0, 0.0, 0.0)
omega0 = np.array([0.0, 0.0, 0.0])

# Define Initial State Vector
x0 = np.concatenate((pos0, vel0, quat0, omega0))

# Simulation Timing
t0 = 0.0
t_end = 5.0
dt = 0.01
n_steps = int(round((t_end - t0) / dt))


def compute_command(t, state, params):
    """
    Cascaded control: reads `state` (position, velocity, quaternion, omega)
    and `t`, and returns [thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd].

    Outer loop (position_controller): position/velocity error -> thrust
    magnitude + desired attitude (the direction the thrust vector needs to
    tilt to move toward target_pos).

    Inner loop (attitude_pd_controller): current attitude/rate error
    relative to that desired attitude -> torque commands.
    """
    pos = state[0:3]
    vel = state[3:6]
    quat = state[6:10]
    omega = state[10:13]

    thrust_cmd, desired_euler = position_controller(
        pos, vel, target_pos, params, Kp_pos, Kd_pos, yaw_des
    )

    tau = attitude_pd_controller(quat, omega, desired_euler, Kp_att, Kd_att)
    roll_cmd, pitch_cmd, yaw_cmd = tau

    return thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd


def step_dynamics(t, state, u, params, dt):
    """
    Advances `state` forward by one control timestep `dt` under a
    constant wrench `u`, using solve_ivp as the sub-stepping integrator.
    """
    sol = solve_ivp(
        fun=lambda t_, x_: quad_dynamics(t_, x_, u, params),
        t_span=(t, t + dt),
        y0=state,
        t_eval=[t + dt],
        method='RK45',
    )
    next_state = sol.y[:, -1]

    # Re-normalize the quaternion. Small numerical drift accumulates over
    # many steps and unit-norm is an assumption the rotation math relies on.
    quat = next_state[6:10]
    next_state[6:10] = quat / np.linalg.norm(quat)

    return next_state


# Initialize logging arrays
time_log = np.zeros(n_steps + 1)
state_log = np.zeros((n_steps + 1, x0.size))   # [pos, vel, quat, omega]
u_log = np.zeros((n_steps, 4))                  # [thrust, roll, pitch, yaw]

# Define the simulation loop.
# NOTE: `state` is the variable that actually advances each iteration.
# `x0` is left untouched as a record of the initial condition -- reusing
# x0 as the running state was the bug in the previous version (it froze
# the sim at t=0 because step_dynamics' return value was never captured).
state = x0.copy()
t = t0
time_log[0] = t0
state_log[0] = state

for i in range(n_steps):
    # Control input from the PD attitude controller
    thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd = compute_command(t, state, params)

    # Mix command down to motors, then back up to the actual wrench
    f, omega = motor_speeds_from_wrench(thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd, params)
    u, f_actual = wrench_from_motor_speeds(omega, params)

    # Integrate the dynamics by one timestep -- capture the returned state
    state = step_dynamics(t, state, u, params, dt)
    t += dt

    # Log
    u_log[i] = u
    time_log[i + 1] = t
    state_log[i + 1] = state

# Extract states for plotting
x_pos, y_pos, z_pos = state_log[:, 0], state_log[:, 1], state_log[:, 2]
quats = state_log[:, 6:10]  # [qx, qy, qz, qw] over time

# Convert quaternion history to Euler angles (deg) to visualize attitude
euler_hist = R.from_quat(quats).as_euler('xyz', degrees=True)
roll_deg = euler_hist[:, 0]
pitch_deg = euler_hist[:, 1]
yaw_deg = euler_hist[:, 2]

# Plot Results
plt.figure(figsize=(12, 7))

plt.subplot(2, 2, 1)
plt.plot(time_log, x_pos, label='x')
plt.plot(time_log, y_pos, label='y')
plt.plot(time_log, z_pos, label='z')
plt.axhline(target_pos[0], color='C0', linewidth=0.75, linestyle='--')
plt.axhline(target_pos[1], color='C1', linewidth=0.75, linestyle='--')
plt.axhline(target_pos[2], color='C2', linewidth=0.75, linestyle='--')
plt.title('Position vs. target (dashed)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(x_pos, y_pos, 'b-', linewidth=2)
plt.plot(target_pos[0], target_pos[1], 'r*', markersize=12, label='target (x,y)')
plt.plot(x_pos[0], y_pos[0], 'go', label='start')
plt.title('Horizontal (X-Y) Path')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.axis('equal')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(time_log, roll_deg, label='roll')
plt.plot(time_log, pitch_deg, label='pitch')
plt.plot(time_log, yaw_deg, label='yaw')
plt.axhline(0, color='k', linewidth=0.75, linestyle='--')
plt.title('Attitude commanded by outer loop')
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(time_log[:-1], u_log[:, 0], 'k-', label='thrust')
plt.title('Commanded Thrust')
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()