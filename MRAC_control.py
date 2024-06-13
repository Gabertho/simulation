import numpy as np
from PID_control import PID_Control

class MRAC_Control(PID_Control):
    def __init__(self):
        PID_Control.__init__(self)
        self.ref_model_states = np.array([[0], [0], [0], [0], [0], [0]])
        self.BW = 2
        self.number_centers = 25
        self.adaptive_gain = 0.01  # 0.01 best for the given disturbance, after that failure

        equal_spacing = np.linspace(-2, 2, self.number_centers)
        self.centers = equal_spacing * np.ones((6, self.number_centers))
        self.basis = np.zeros((self.number_centers, 1))
        self.output_weight = np.zeros((self.number_centers, 3))

    def reference_model(self, yaw_pitch_roll_ref_OL):
        yaw_OL, pitch_OL, roll_OL = yaw_pitch_roll_ref_OL[0], yaw_pitch_roll_ref_OL[1], yaw_pitch_roll_ref_OL[2]
        wn = 20
        damping = 0.1
        Arm = np.array([[0, 1, 0, 0, 0, 0],
                        [-wn**2, -2 * damping * wn, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, -wn**2, -2 * damping * wn, 0, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, -wn**2, -2 * damping * wn]])

        Brm = np.array([[0, 0, 0], [wn**2, 0, 0], [0, 0, 0],
                        [0, wn**2, 0], [0, 0, 0], [0, 0, wn**2]])

        ref = np.array([[roll_OL], [pitch_OL], [yaw_OL]])

        Bm_rt_pdt = np.dot(Brm, ref)

        k1 = np.dot(Arm, self.ref_model_states) + Bm_rt_pdt
        k2 = np.dot(Arm, (self.ref_model_states + k1 * self.delt / 2)) + Bm_rt_pdt
        k3 = np.dot(Arm, (self.ref_model_states + k2 * self.delt / 2)) + Bm_rt_pdt
        k4 = np.dot(Arm, (self.ref_model_states + k3 * self.delt)) + Bm_rt_pdt

        self.ref_model_states = self.ref_model_states + (self.delt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  # roll, d_roll, pitch, d_pitch, yaw, d_yaw

    def mrac_weight_update(self, ref_model_states):
        current_rpy_state = np.array([[self.roll], [self.D_roll], [self.pitch], [self.D_pitch], [self.yaw], [self.D_yaw]])

        for i in range(0, self.number_centers):
            expression_1 = self.centers[:, i].reshape(-1, 1) - current_rpy_state
            expression = (-(np.linalg.norm(expression_1)**2)) / (2 * self.BW)
            self.basis[i] = np.exp(expression)

        self.basis[0] = 1

        error = ref_model_states - current_rpy_state
        P = np.array([[50.13, 0.0013, 0, 0, 0, 0], [0.0013, 0.1253, 0, 0, 0, 0],
                      [0, 0, 50.13, 0.0013, 0, 0], [0, 0, 0.0013, 0.1253, 0, 0],
                      [0, 0, 0, 0, 50.13, 0.0013], [0, 0, 0, 0, 0.0013, 0.1253]])

        B = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0],
                      [0, 1, 0], [0, 0, 0], [0, 0, 1]])

        self.output_weight = self.output_weight + (-self.delt) * (self.adaptive_gain) * np.dot(self.basis, np.dot(error.T, np.dot(P, B)))

    def mrac_torque(self):
        vad = np.dot(self.output_weight.T, self.basis)
        u_net = -vad
        return u_net
