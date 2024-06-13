
import math

class PID_Control():
    def __init__(self):
        # state_vec is (X, Y, Z, yaw, pitch, roll, dx, dy, dz, p, q, r)
        # (x, y, z, yaw, pitch, roll, dx, dy, dz) in earth/inertial frame
        # (p, q, r) in body frame
        self.delt = 0.005  # time step
        self.mass = 0.063
        self.g = 9.81
        
        # Values for Parrot Mambo 
        
        self.P_yaw = 0.004  
        self.D_yaw = 0.3 * 0.004
        
        self.P_pitch = 0.013
        self.I_pitch = 0.01
        self.D_pitch = 0.002
        
        self.P_roll = 0.01
        self.I_roll = 0.01
        self.D_roll = 0.0028
        
        self.P_x = -0.44
        self.I_x = 0
        self.D_x = -0.35
        
        self.P_y = -0.44  
        self.I_y = 0  
        self.D_y = -0.35 
        
        self.P_z = 0.8
        self.D_z = 0.3
        
        self.integration_val_pitch = 0
        self.integration_val_roll = 0
        self.integration_val_x = 0
        self.integration_val_y = 0

    def assign_states(self, state):
        self.X, self.Y, self.Z = state[0], state[1], state[2]
        self.yaw, self.pitch, self.roll = state[3], state[4], state[5]
        self.dx, self.dy, self.dz = state[6], state[7], state[8]
        self.p, self.q, self.r = state[9], state[10], state[11]

    def integral(self, prev_integral_value, error):
        # 0.001 is antiwindup gain
        # if prev_integral_value >= 2:  # For modeling saturation
        #     prev_integral_value = 2
        # elif prev_integral_value <= -2:
        #     prev_integral_value = -2
        error_anti_windup = error - 0.001 * prev_integral_value
        integral_new = prev_integral_value + error_anti_windup * self.delt
        return integral_new

    def PID_control(self, Kp, Ki, Kd, error, int_error, d_error):
        PID_control_value = Kp * error + Ki * int_error - Kd * d_error
        return PID_control_value

    def outer_loop_control(self, ref_traj):
        # ref_traj is (Xref, Yref, Zref, Yaw_ref)
        Xref, Yref, Yaw_ref = ref_traj[0], ref_traj[1], ref_traj[3]
        
        error_x = Xref - self.X
        self.integration_val_x = self.integral(self.integration_val_x, error_x)
        
        error_y = Yref - self.Y
        self.integration_val_y = self.integral(self.integration_val_y, error_y)
        
        PID_x = self.PID_control(self.P_x, self.I_x, self.D_x, error_x, self.integration_val_x, self.dx)
        PID_y = self.PID_control(self.P_y, self.I_y, self.D_y, error_y, self.integration_val_y, self.dy)
        
        pitch_ref = PID_x * math.cos(self.yaw) + PID_y * math.sin(self.yaw)  # Approx Model Inversion
        roll_ref = PID_x * math.sin(self.yaw) - PID_y * math.cos(self.yaw)  # Approx Model Inversion
        
        return [Yaw_ref, pitch_ref, roll_ref]

    def Thrust_force(self, ref_traj):
        Zref = ref_traj[2]
        error_z = Zref - self.Z
        
        Thrust_force_PID = self.PID_control(self.P_z, 0, self.D_z, error_z, 0, self.dz)
        # Thrust_force_total = (-self.mass * self.g + Thrust_force_PID) / (math.cos(self.pitch) * math.cos(self.roll))
        
        return Thrust_force_PID

    def inner_loop_control(self, yaw_pitch_roll_ref):
        error_yaw = yaw_pitch_roll_ref[0] - self.yaw
        
        error_pitch = yaw_pitch_roll_ref[1] - self.pitch
        self.integration_val_pitch = self.integral(self.integration_val_pitch, error_pitch)
        
        error_roll = yaw_pitch_roll_ref[2] - self.roll
        self.integration_val_roll = self.integral(self.integration_val_roll, error_roll)
        
        torque_yaw = self.PID_control(self.P_yaw, 0, self.D_yaw, error_yaw, 0, self.r)
        torque_pitch = self.PID_control(self.P_pitch, self.I_pitch, self.D_pitch, error_pitch, self.integration_val_pitch, self.q)
        torque_roll = self.PID_control(self.P_roll, self.I_roll, self.D_roll, error_roll, self.integration_val_roll, self.p)
        
        return [torque_roll, torque_pitch, torque_yaw]
