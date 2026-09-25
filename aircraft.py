import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class aircraft():
    def __init__(self, mass=10000.0, Ix=15000.0, Iy=50000.0, Iz=60000.0, g=9.8):

        self.m = mass  
        self.Ix = Ix  # моменты инерции [кг/м**2]
        self.Iy = Iy  
        self.Iz = Iz  
        self.g = g 

        self.t = None
        self.sol = None     

    def model(self, t, state):
        x, y, z, vx, vy, vz, psi, theta, gamma, wx, wy, wz = state 

        # своровано с умной нейронки
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        alpha = np.clip(np.arctan2(vy, vx), -0.25, 0.25)
        beta  = np.clip(np.arcsin(np.clip(vz / v, -1.0, 1.0)), -0.25, 0.25)
        Mx = -15000.0 * wx - 20000.0 * beta
        My = -40000.0 * wy - 60000.0 * alpha
        Mz = -20000.0 * wz + 15000.0 * beta
        T = 7000.0
        air_res = 0.012 * self.m * self.g * (v / 120.0)**2
        Fx = T - air_res + self.m * self.g * np.sin(theta) 
        Fy = -self.m * self.g * np.cos(theta) * np.cos(gamma) + self.m * self.g * (0.95 + 5.0 * alpha)
        Fz =  self.m * self.g * np.cos(theta) * np.sin(gamma) - 2.0 * self.m * self.g * beta
        
        # своровано с какой-то методички из интернета
        dxdt = vx * np.cos(theta) * np.cos(psi) + vy * (np.sin(gamma) * np.sin(psi) - np.cos(gamma) * np.sin(theta) * np.cos(psi)) + vz * (np.cos(gamma) * np.sin(psi) + np.sin(gamma) * np.sin(theta) * np.cos(psi))
        dydt = vx * np.sin(theta) + vy * np.cos(gamma) * np.cos(theta) - vz * (np.sin(gamma) * np.cos(theta))
        dzdt = -vx * np.cos(theta) * np.sin(psi) + vy * (np.sin(gamma) * np.cos(psi) + np.cos(gamma) * np.sin(theta) * np.sin(psi)) + vz * (np.cos(gamma) * np.cos(psi) - np.sin(gamma) * np.sin(theta) * np.sin(psi))
        dvxdt = Fx/self.m + vy * wz - vz *wy
        dvydt = Fy/self.m + vz* wx - vx * wz
        dvzdt = Fz/self.m + vx * wy - vy * wx
        dpsidt = 1/np.cos(theta) * (wy * np.cos(gamma) - wz * np.sin(gamma))
        dthetadt = wy * np.sin(gamma) + wz * np.cos(gamma)
        dgammadt = wx - np.tan(theta) * (wy * np.cos(gamma) - wz * np.sin(gamma))
        dwxdt = 1/self.Ix * (Mx + (self.Iy - self.Iz) * wy * wz)
        dwydt = 1/self.Iy * (My + (self.Iz - self.Ix) * wz * wx)
        dwzdt = 1/self.Iz * (Mz + (self.Ix - self.Iy) * wx * wy)

        return [dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt, dpsidt, dthetadt, dgammadt, dwxdt, dwydt, dwzdt]

    def integrate(self, y0, t_span):

        sol = solve_ivp(
            fun=self.model,
            t_span=t_span,
            y0=y0,
            t_eval=None,  
            rtol=1e-6,
            atol=1e-8,
            method='RK45',
            dense_output=True
        )

        self.t = sol.t
        self.sol = sol.y      
        self.raw_sol = sol
        return sol

    def plot_motion(self):

        t = self.t
        x, y, z = self.sol[0], self.sol[1], self.sol[2]
        
        fig = plt.figure(figsize=(12, 8))

        # 3d
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(x, z, y, 'b-', linewidth=1.5)
        ax1.scatter(x[0], z[0], y[0], c='g', s=50, label='Старт')
        ax1.scatter(x[-1], z[-1], y[-1], c='r', s=50, label='Финиш')
        ax1.set_xlabel('X [м]')
        ax1.set_ylabel('Z [м]')
        ax1.set_zlabel('Y [м]')

        ax1.grid(True)
        # проекции
        ax2 = fig.add_subplot(2, 3, 5)
        ax2.plot(x, z, 'b-')
        ax2.scatter(x[0], z[0], c='g', s=50)
        ax2.scatter(x[-1], z[-1], c='r', s=50)
        ax2.set_xlabel('X_g [м]')
        ax2.set_ylabel('Z_g [м]')
        ax2.axis('equal')

        ax2.grid(True)

        ax3 = fig.add_subplot(2, 3, 4)
        ax3.plot(x, y, 'b-')
        ax3.scatter(x[0], y[0], c='g', s=50)
        ax3.scatter(x[-1], y[-1], c='r', s=50)
        ax3.set_xlabel('X_g [м]')
        ax3.set_ylabel('Y_g [м]')
        ax3.axis('equal')

        ax3.grid(True)

        ax4 = fig.add_subplot(2, 3, 6)
        ax4.plot(z, y, 'b-')
        ax4.scatter(z[0], y[0], c='g', s=50)
        ax4.scatter(z[-1], y[-1], c='r', s=50)
        ax4.set_xlabel('Z_g [м]')
        ax4.set_ylabel('Y_g [м]')
        ax4.axis('equal')

        ax4.grid(True)

        plt.tight_layout()

        plt.show()

        return fig


    def observe(self, target_pos_array, t_eval):
        pass

        
    def jacobian(self):
        x, y, z, vx, vy, vz, psi, theta, gamma, wx, wy, wz = self.sol[0][0], self.sol[1][0], self.sol[2][0], self.sol[3][0], self.sol[4][0], self.sol[5][0], self.sol[6][0], self.sol[7][0], self.sol[8][0], self.sol[9][0], self.sol[10][0], self.sol[11][0]
        A1_1 =  0
        A1_2 =  0
        A1_3 =  0
        A1_4 =  np.cos(psi) * np.cos(theta)
        A1_5 =  np.sin(gamma)*np.sin(psi) - np.sin(theta)*np.cos(gamma)*np.cos(psi)
        A1_6 =  np.sin(gamma)*np.sin(theta)*np.cos(psi) + np.sin(psi)*np.cos(gamma)
        A1_7 =  -vx*np.sin(psi)*np.cos(theta) + vy*(np.sin(gamma)*np.cos(psi) + np.sin(psi)*np.sin(theta)*np.cos(gamma)) + vz*(-np.sin(gamma)*np.sin(psi)*np.sin(theta)+ np.cos(gamma)*np.cos(psi))
        A1_8 =  -vx*np.sin(theta)*np.cos(psi) - vy*np.cos(gamma)*np.cos(psi)*np.cos(theta) + vz*np.sin(gamma)*np.cos(psi)*np.cos(theta)
        A1_9 =  vy*(np.sin(gamma)*np.sin(theta)*np.cos(psi) + np.sin(psi)*np.cos(gamma)) + vz*(-np.sin(gamma)*np.sin(psi) + np.sin(theta)*np.cos(gamma)*np.cos(psi))
        A1_10 =  0
        A1_11 =  0
        A1_12 =  0
        A2_1 =  0
        A2_2 =  0
        A2_3 =  0
        A2_4 =  np.sin(theta)
        A2_5 =  np.cos(gamma)*np.cos(theta)
        A2_6 =  -np.sin(gamma)*np.cos(theta)
        A2_7 =  0
        A2_8 =  vx*np.cos(theta) - vy*np.sin(theta)*np.cos(gamma) + vz*np.sin(gamma)*np.sin(theta)
        A2_9 =  -vy*np.sin(gamma)*np.cos(theta) - vz*np.cos(gamma)*np.cos(theta)
        A2_10 =  0
        A2_11 =  0
        A2_12 =  0
        A3_1 =  0
        A3_2 =  0
        A3_3 =  0
        A3_4 =  -np.sin(psi)*np.cos(theta)
        A3_5 =  np.sin(gamma)*np.cos(psi) + np.sin(psi)*np.sin(theta)*np.cos(gamma)
        A3_6 =  -np.sin(gamma)*np.sin(psi)*np.sin(theta) + np.cos(gamma)*np.cos(psi)
        A3_7 =  -vx*np.cos(psi)*np.cos(theta) + vy*(-np.sin(gamma)*np.sin(psi) + np.sin(theta)*np.cos(gamma)*np.cos(psi)) + vz*(-np.sin(gamma)*np.sin(theta)*np.cos(psi) - np.sin(psi)*np.cos(gamma))
        A3_8 =  vx*np.sin(psi)*np.sin(theta) + vy*np.sin(psi)*np.cos(gamma)*np.cos(theta) - vz*np.sin(gamma)*np.sin(psi)*np.cos(theta)
        A3_9 =  vy*(-np.sin(gamma)*np.sin(psi)*np.sin(theta) + np.cos(gamma)*np.cos(psi)) + vz*(-np.sin(gamma)*np.cos(psi) - np.sin(psi)*np.sin(theta)*np.cos(gamma))
        A3_10 =  0
        A3_11 =  0
        A3_12 =  0
        A4_1 =  0
        A4_2 =  0
        A4_3 =  0
        A4_4 =  0
        A4_5 =  wz
        A4_6 =  -wy
        A4_7 =  0
        A4_8 =  0
        A4_9 =  0
        A4_10 =  0
        A4_11 =  -vz
        A4_12 =  vy
        A5_1 =  0
        A5_2 =  0
        A5_3 =  0
        A5_4 =  -wz
        A5_5 =  0
        A5_6 =  wx
        A5_7 =  0
        A5_8 =  0
        A5_9 =  0
        A5_10 =  vz
        A5_11 =  0
        A5_12 =  -vx
        A6_1 =  0
        A6_2 =  0
        A6_3 =  0
        A6_4 =  wy
        A6_5 =  -wx
        A6_6 =  0
        A6_7 =  0
        A6_8 =  0
        A6_9 =  0
        A6_10 =  -vy
        A6_11 =  vx
        A6_12 =  0
        A7_1 =  0
        A7_2 =  0
        A7_3 =  0
        A7_4 =  0
        A7_5 =  0
        A7_6 =  0
        A7_7 =  0
        A7_8 =  (wy*np.cos(gamma) - wz*np.sin(gamma))*np.sin(theta)/np.cos(theta)**2
        A7_9 =  (-wy*np.sin(gamma) - wz*np.cos(gamma))/np.cos(theta)
        A7_10 =  0
        A7_11 =  np.cos(gamma)/np.cos(theta)
        A7_12 =  -np.sin(gamma)/np.cos(theta)
        A8_1 =  0
        A8_2 =  0
        A8_3 =  0
        A8_4 =  0
        A8_5 =  0
        A8_6 =  0
        A8_7 =  0
        A8_8 =  0
        A8_9 =  wy*np.cos(gamma) - wz*np.sin(gamma)
        A8_10 =  0
        A8_11 =  np.sin(gamma)
        A8_12 =  np.cos(gamma)
        A9_1 =  0
        A9_2 =  0
        A9_3 =  0
        A9_4 =  0
        A9_5 =  0
        A9_6 =  0
        A9_7 =  0
        A9_8 =  -(wy*np.cos(gamma) - wz*np.sin(gamma))*(np.tan(theta)**2 + 1)
        A9_9 =  -(-wy*np.sin(gamma) - wz*np.cos(gamma))*np.tan(theta)
        A9_10 =  1
        A9_11 =  -np.cos(gamma)*np.tan(theta)
        A9_12 =  np.sin(gamma)*np.tan(theta)
        A10_1 =  0
        A10_2 =  0
        A10_3 =  0
        A10_4 =  0
        A10_5 =  0
        A10_6 =  0
        A10_7 =  0
        A10_8 =  0
        A10_9 =  0
        A10_10 =  0
        A10_11 =  wz*(self.Iy - self.Iz)/self.Ix
        A10_12 =  wy*(self.Iy - self.Iz)/self.Ix
        A11_1 =  0
        A11_2 =  0
        A11_3 =  0
        A11_4 =  0
        A11_5 =  0
        A11_6 =  0
        A11_7 =  0
        A11_8 =  0
        A11_9 =  0
        A11_10 =  wz*(-self.Ix + self.Iz)/self.Iy
        A11_11 =  0
        A11_12 =  wx*(-self.Ix + self.Iz)/self.Iy
        A12_1 =  0
        A12_2 =  0
        A12_3 =  0
        A12_4 =  0
        A12_5 =  0
        A12_6 =  0
        A12_7 =  0
        A12_8 =  0
        A12_9 =  0
        A12_10 =  wy*(self.Ix - self.Iy)/self.Iz
        A12_11 =  wx*(self.Ix - self.Iy)/self.Iz
        A12_12 =  0       

        A_1 = [A1_1, A1_2, A1_3, A1_4, A1_5, A1_6, A1_7, A1_8, A1_9, A1_10, A1_11, A1_12]
        A_2 = [A2_1, A2_2, A2_3, A2_4, A2_5, A2_6, A2_7, A2_8, A2_9, A2_10, A2_11, A2_12]
        A_3 = [A3_1, A3_2, A3_3, A3_4, A3_5, A3_6, A3_7, A3_8, A3_9, A3_10, A3_11, A3_12]
        A_4 = [A4_1, A4_2, A4_3, A4_4, A4_5, A4_6, A4_7, A4_8, A4_9, A4_10, A4_11, A4_12]
        A_5 = [A5_1, A5_2, A5_3, A5_4, A5_5, A5_6, A5_7, A5_8, A5_9, A5_10, A5_11, A5_12]
        A_6 = [A6_1, A6_2, A6_3, A6_4, A6_5, A6_6, A6_7, A6_8, A6_9, A6_10, A6_11, A6_12]
        A_7 = [A7_1, A7_2, A7_3, A7_4, A7_5, A7_6, A7_7, A7_8, A7_9, A7_10, A7_11, A7_12]
        A_8 = [A8_1, A8_2, A8_3, A8_4, A8_5, A8_6, A8_7, A8_8, A8_9, A8_10, A8_11, A8_12]
        A_9 = [A9_1, A9_2, A9_3, A9_4, A9_5, A9_6, A9_7, A9_8, A9_9, A9_10, A9_11, A9_12]
        A_10 = [A10_1, A10_2, A10_3, A10_4, A10_5, A10_6, A10_7, A10_8, A1_9, A10_10, A10_11, A10_12]
        A_11 = [A11_1, A11_2, A11_3, A11_4, A11_5, A11_6, A11_7, A11_8, A1_9, A11_10, A11_11, A11_12]
        A_12 = [A12_1, A12_2, A12_3, A12_4, A12_5, A12_6, A12_7, A12_8, A1_9, A12_10, A12_11, A12_12]

        A = [A_1, A_2, A_3, A_4, A_5, A_6, A_7, A_8, A_9, A_10, A_11, A_12]

        return A
