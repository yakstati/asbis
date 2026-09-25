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