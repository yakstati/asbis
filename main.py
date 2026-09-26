# закинуть фк в интегрирование, прикрутить измерения до цели (на основе x_o, p_o), квантование (так же на основе фк), пораскинуть извилиной над прицеливанием

from target import target
from aircraft import aircraft
import numpy as np
import matplotlib.pyplot as plt
from plot_fk import plot_fk

ac = aircraft()
tar = target(pos=[2000.0, 0.0, 0.0], vel=[-16.6, 0.0, 0.0])

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

sol = ac.integrate(y0, t_span, t_eval)
sol_tar = tar.integrate(t_span)

if not sol.success:
    print("ошибка интегрирования движения ла", sol.message)

if not sol_tar.success:
    print("ошибка интегрирования движения цели", sol.message)

P_o, X_o = ac.fk(t_eval)
ac.observe(sol_tar, t_eval)

plot_fk(ac, X_o, P_o, t_eval)

plt.show()
