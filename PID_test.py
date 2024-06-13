from PID_control import PID_Control
import matplotlib.pyplot as plt
import numpy as np
import math

def drone_dynamics_with_control():
    A = PID_Control()
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

    T_fig_8 = 20

    for time in range(1, 5000):  # Number of time steps
        ref_traj = [math.cos(2 * math.pi * time * 0.005 / T_fig_8),
                    math.sin(4 * math.pi * time * 0.005 / T_fig_8), -2, 0]  # Xref(t), Yref(t), Zref(t), Yawref(t)
        
        [Yaw_ref, pitch_ref, roll_ref] = A.outer_loop_control(ref_traj)
        Thrust = A.Thrust_force(ref_traj)
        [torque_roll, torque_pitch, torque_yaw] = A.inner_loop_control([Yaw_ref, pitch_ref, roll_ref])

        theta = A.pitch
        psi = A.yaw
        phi = A.roll

        first_row = [math.cos(theta) * math.cos(psi),
                     math.sin(phi) * math.sin(theta) * math.cos(psi) - math.cos(phi) * math.sin(psi),
                     math.cos(phi) * math.sin(theta) * math.cos(psi) + math.sin(phi) * math.sin(psi)]

        second_row = [math.cos(theta) * math.sin(psi),
                      math.sin(phi) * math.sin(theta) * math.sin(psi) + math.cos(phi) * math.cos(psi),
                      math.cos(phi) * math.sin(theta) * math.sin(psi) - math.sin(phi) * math.cos(psi)]

        third_row = [-math.sin(theta),
                     math.sin(phi) * math.cos(theta),
                     math.cos(phi) * math.cos(theta)]

        R_body_to_veh = np.array([first_row, second_row, third_row])

        # x_(t+1) = f(x_t, u_t)
        [X_next] = np.array(A.X) + time_step * (np.dot(first_row, [[u], [v], [w]]))
        [Y_next] = np.array(A.Y) + time_step * (np.dot(second_row, [[u], [v], [w]]))
        [Z_next] = np.array(A.Z) + time_step * (np.dot(third_row, [[u], [v], [w]]))

        u_next = u + time_step * (A.r * v - A.q * w - 9.81 * math.sin(theta))
        v_next = v + time_step * (A.p * w - A.r * u + 9.81 * math.cos(theta) * math.sin(phi))
        w_next = w + time_step * (A.q * u - A.p * v + 9.81 * math.cos(theta) * math.cos(phi) + Thrust / A.mass)

        phi_next = phi + time_step * (A.p + A.q * math.sin(phi) * math.tan(theta) + A.r * math.cos(phi) * math.tan(theta))
        theta_next = theta + time_step * (A.q * math.cos(phi) - A.r * math.sin(phi))
        psi_next = psi + time_step * ((A.q * math.sin(phi) + A.r * math.cos(phi)) / math.cos(theta))

        p_next = A.p + time_step * (((Jy - Jz) / Jx) * A.q * A.r + torque_roll / Jx)
        q_next = A.q + time_step * (((Jz - Jx) / Jy) * A.p * A.r + torque_pitch / Jy)
        r_next = A.r + time_step * (((Jx - Jy) / Jz) * A.p * A.q + torque_yaw / Jz)

        [dx_next], [dy_next], [dz_next] = np.dot(R_body_to_veh, [[u_next], [v_next], [w_next]])

        u, v, w = u_next, v_next, w_next
        A.assign_states([X_next, Y_next, Z_next, psi_next, theta_next, phi_next, dx_next, dy_next, dz_next, p_next, q_next, r_next])

        X_vals.append(A.X)
        Y_vals.append(A.Y)
        Z_vals.append(A.Z)
        ref_traj_x.append(ref_traj[0])
        ref_traj_y.append(ref_traj[1])

    plt.plot(Y_vals, X_vals)
    plt.plot(ref_traj_y, ref_traj_x, 'k')
    plt.show()

drone_dynamics_with_control()
