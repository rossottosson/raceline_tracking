import numpy as np
from numpy.typing import ArrayLike

# --- Global State Variables ---
v_integral = 0.0
last_v_error = 0.0
last_delta_error = 0.0
last_closest_idx = 0 

# --- System Time Step ---
H = 0.1 

# PID / Steering
K_P_V = 8.0
K_I_V = 3.0
K_D_V = 0.5
INTEGRAL_WINDUP_LIMIT = 20.0

# Final Tuned Parameters
K_P_DELTA = 8.0       # Steering Gain
K_D_DELTA = 0.3        # Steering Damping
LOOKAHEAD_BASE = 10.0   # Short lookahead for tight corners
LOOKAHEAD_TIME = 0.35  # Look ahead 0.35s based on speed
FRICTION_LIMIT_G = 1.4 # Cornering Grip (~1.4G is typical for this car)
MAX_VELOCITY = 90.0    # Top Speed on straights
BRAKING_AGGRESSION = 20.0 # m/s^2 (Average braking capability)

def reset_controller_state():
    """Resets the controller's internal state (integrals, etc.)"""
    global v_integral, last_v_error, last_delta_error, last_closest_idx
    v_integral = 0.0
    last_v_error = 0.0
    last_delta_error = 0.0
    last_closest_idx = 0

def lower_controller(state : ArrayLike, desired : ArrayLike, parameters : ArrayLike) -> ArrayLike:
    """Low-level actuation (Throttle/Steering Rate)"""
    global v_integral, last_v_error, last_delta_error
    
    current_steer = state[2]
    current_v = state[3]
    desired_steer = desired[0]
    desired_v = desired[1]

    # C1: Longitudinal PID
    error_v = desired_v - current_v
    p_v = K_P_V * error_v
    v_integral = np.clip(v_integral + error_v * H, -INTEGRAL_WINDUP_LIMIT, INTEGRAL_WINDUP_LIMIT)
    i_v = K_I_V * v_integral
    d_v = K_D_V * (error_v - last_v_error) / H
    a = p_v + i_v + d_v
    last_v_error = error_v

    # C2: Lateral PD
    error_delta = desired_steer - current_steer
    # Normalize angle to [-pi, pi]
    error_delta = np.arctan2(np.sin(error_delta), np.cos(error_delta))
    
    p_delta = K_P_DELTA * error_delta
    d_delta = K_D_DELTA * (error_delta - last_delta_error) / H
    v_delta = p_delta + d_delta
    last_delta_error = error_delta
    
    return np.array([v_delta, a]).T

def controller(state : ArrayLike, parameters : ArrayLike, racetrack) -> ArrayLike:
    """
    Computes desired steering and velocity using Kinematic Velocity Profiling.
    """
    global last_closest_idx
    
    car_pos = state[0:2]
    current_v = state[3]
    current_heading = state[4]
    wheelbase = parameters[0]

    path = racetrack.centerline
    n_points = len(path)
    
    # --- 1. Robust Index Finding & Lateral Error ---
    search_window = 100 
    start_search = last_closest_idx
    end_search = last_closest_idx + search_window
    
    indices = np.arange(start_search, end_search) % n_points
    distances = np.linalg.norm(path[indices] - car_pos, axis=1)
    
    current_local_idx = np.argmin(distances)
    closest_idx = indices[current_local_idx]
    last_closest_idx = closest_idx

    lateral_error = distances[current_local_idx] 
    
    # --- 2. Steering Logic (Pure Pursuit) ---
    steer_dist = (LOOKAHEAD_TIME * current_v) + LOOKAHEAD_BASE
    
    steer_idx = closest_idx
    dist_accum = 0.0
    while dist_accum < steer_dist:
        curr = path[steer_idx % n_points]
        next_p = path[(steer_idx + 1) % n_points]
        dist_accum += np.linalg.norm(next_p - curr)
        steer_idx += 1
    
    steer_target = path[steer_idx % n_points]
    
    vec_steer = steer_target - car_pos
    alpha_steer = np.arctan2(vec_steer[1], vec_steer[0])
    err_steer = np.arctan2(np.sin(alpha_steer - current_heading), np.cos(alpha_steer - current_heading))
    
    desired_steer_angle = np.arctan(2.0 * wheelbase * np.sin(err_steer) / np.linalg.norm(vec_steer))

    # --- 3. Velocity Logic (Kinematic Profiling) ---
    LOOKAHEAD_DISTANCE = 200.0 
    current_scan_dist = 0.0
    scan_idx = closest_idx
    
    final_target_velocity = MAX_VELOCITY
    friction_limit = 9.81 * FRICTION_LIMIT_G
    STEP = 2 
    
    while current_scan_dist < LOOKAHEAD_DISTANCE:
        idx_curr = scan_idx % n_points
        idx_p1 = idx_curr
        idx_p2 = (idx_curr + STEP) % n_points
        idx_p3 = (idx_curr + 2 * STEP) % n_points

        p1 = path[idx_p1]
        p2 = path[idx_p2]
        p3 = path[idx_p3]
        
        segment_len = np.linalg.norm(path[(scan_idx + 1) % n_points] - path[scan_idx % n_points])
        current_scan_dist += segment_len
        
        # --- Curvature Calculation ---
        v1 = p2 - p1
        v2 = p3 - p2
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        dot = np.dot(v1, v2)

        if norm_v1 * norm_v2 > 1e-6:
            angle = np.arccos(np.clip(dot / (norm_v1 * norm_v2), -1.0, 1.0))
        else:
            angle = 0.0
            
        if angle > 1e-4:
            radius = norm_v1 / angle
            v_phys = np.sqrt(friction_limit * radius)
            v_corner_limit = max(v_phys, 8.0) 
        else:
            v_corner_limit = MAX_VELOCITY

        # KINEMATIC LIMIT
        v_allowable = np.sqrt(v_corner_limit**2 + 2.0 * BRAKING_AGGRESSION * current_scan_dist)
        
        if v_allowable < final_target_velocity:
            final_target_velocity = v_allowable
            
        scan_idx += 1

        # --- 4. Safety Governor ---
    if lateral_error > 0.6: 
        scaling_factor = 1.0 / (1.0 + ((lateral_error - 0.6)))
        final_target_velocity *= scaling_factor

    # --- 4. Return ---
    return np.array([desired_steer_angle, final_target_velocity]).T
