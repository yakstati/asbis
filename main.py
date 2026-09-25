# 

from aircraft import aircraft
import numpy as np

ac = aircraft()

y0 = [
        0.0,      # x_g
        2000.0,   # y_g 
        0.0,      # z_g
        120.0,    # Vx
        0.0,      # Vy 
        0.0,      # Vz
        0.0,      # psi
        0.0,      # theta
        0.0,      # gamma
        0.0,      # wx
        0.0,      # wy
        0.0       # wz
]


t_span = (0.0, 90.0)
t_eval = np.linspace(0.0, 90.0, 901)

sol = ac.integrate(y0, t_span)

if not sol.success:
    print("ошибка интегрирования", sol.message)

ac.plot_motion()