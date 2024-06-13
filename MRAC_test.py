# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:57:34 2020

@author: me112

Test Script
"""
from MRAC_control import MRAC_Control
import matplotlib.pyplot as plt
import numpy as np
import math

def drone_dynamics_with_control():
    A = MRAC_Control()
    A.assign_states([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    time_step = A.delt
    u, v, w = 0, 0, 0
    Jx = 0.0000582857
    Jy = 0.0000716914
    Jz = 0.0001

    X_vals = []
    Y_vals = []
    Z_vals = []
    ref_traj_x = []
    ref_traj_y = []
    roll_vals = []
    roll_refm_list = []
    roll_OL_list = []

    T_fig_8 = 20

    for time in range(1, 5000):
        ref_traj = [math.cos(2 * math.pi * time * 0.005 / T_fig_8), math.sin(4 * math.pi * time * 0.005 / T_fig_8), -2, 0]

        [yaw_ref_OL, pitch_ref_OL, roll_ref_OL] = A.outer_loop_control(ref_traj)

        Thrust = A.Thrust_force(ref_traj)

        [refm_roll], [refm_pitch], [refm_yaw] = A.ref_model_states[0], A.ref_model_states[2], A.ref_model_states[4]

        [torque_roll_pid, torque_pitch_pid, torque_yaw_pid] = A.inner_loop_control([refm_yaw, refm_pitch, refm_roll])

        [torque_roll_ad], [torque_pitch_ad], [torque_yaw_ad] = A.mrac_torque()

        torque_roll = torque_roll_pid + torque_roll_ad
        torque_pitch = torque_pitch_pid + torque_pitch_ad
        torque_yaw = torque_yaw_pid + torque_yaw_ad

        theta = A.pitch
        psi = A.yaw
        phi = A.roll

        first_row = [
            math.cos(theta) * math.cos(psi),
            math.sin(phi) * math.sin(theta) * math.cos(psi) - math.cos(phi) * math.sin(psi),
            math.cos(phi) * math.sin(theta) * math.cos(psi) + math.sin(phi) * math.sin(psi)
        ]

        second_row = [
            math.cos(theta) * math.sin(psi),
            math.sin(phi) * math.sin(theta) * math.sin(psi) + math.cos(phi) * math.cos(psi),
            math.cos(phi) * math.sin(theta) * math.sin(psi) - math.sin(phi) * math.cos(psi)
        ]

        third_row = [
            -math.sin(theta),
            math.sin(phi) * math.cos(theta),
            math.cos(phi) * math.cos(theta)
        ]

        R_body_to_veh = np.array([first_row, second_row, third_row])

        [X_next] = np.array(A.X) + time_step * (np.dot(first_row, [[u], [v], [w]]))
        [Y_next] = np.array(A.Y) + time_step * (np.dot(second_row, [[u], [v], [w]]))
        [Z_next] = np.array(A.Z) + time_step * (np.dot(third_row, [[u], [v], [w]]))

        u_next = u + time_step * (A.r * v - A.q * w - 9.81 * math.sin(theta))
        v_next = v + time_step * (A.p * w - A.r * u + 9.81 * math.cos(theta) * math.sin(phi))
        w_next = w + time_step * (A.q * u - A.p * v + 9.81 * math.cos(theta) * math.cos(phi) + Thrust / A.mass)

        phi_next = phi + time_step * (A.p + A.q * math.sin(phi) * math.tan(theta) + A.r * math.cos(phi) * math.tan(theta))
        theta_next = theta + time_step * (A.q * math.cos(phi) - A.r * math.sin(phi))
        psi_next = psi + time_step * ((A.q * math.sin(phi) + A.r * math.cos(phi)) / math.cos(theta))

        if A.X > 0:
            wind_x = 0
            wind_y = 0
        else:
            wind_x = 0
            wind_y = 0

        p_next = A.p + time_step * (((Jy - Jz) / Jx) * A.q * A.r + torque_roll / Jx) + time_step * wind_x
        q_next = A.q + time_step * (((Jz - Jx) / Jy) * A.p * A.r + torque_pitch / Jy) + time_step * wind_y
        r_next = A.r + time_step * (((Jx - Jy) / Jz) * A.p * A.q + torque_yaw / Jz)

        [dx_next], [dy_next], [dz_next] = np.dot(R_body_to_veh, [[u_next], [v_next], [w_next]])

        u, v, w = u_next, v_next, w_next

        A.assign_states([X_next, Y_next, Z_next, psi_next, theta_next, phi_next, dx_next, dy_next, dz_next, p_next, q_next, r_next])

        A.reference_model([yaw_ref_OL, pitch_ref_OL, roll_ref_OL])

        A.mrac_weight_update(A.ref_model_states)

        X_vals.append(A.X)
        Y_vals.append(A.Y)
        Z_vals.append(A.Z)

        roll_OL_list.append(roll_ref_OL)
        roll_vals.append(A.roll)
        roll_refm_list.append(refm_roll)

        ref_traj_x.append(ref_traj[0])
        ref_traj_y.append(ref_traj[1])

    plt.plot(Y_vals, X_vals)
    plt.plot(ref_traj_y, ref_traj_x, 'k')
    plt.show()

drone_dynamics_with_control()
