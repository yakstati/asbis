# фк по x отрабатывает нормально в рамках положению по x, для улучшения дополнить якобиан с учетом зависимости F и M
# поразмыслить над СУ
from target import target
from aircraft import aircraft
import numpy as np
import matplotlib.pyplot as plt

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



#print(X_o[:, 0])

t = ac.t
x = ac.sol[0]
#print('t=', t)
#print('x=', x)
#print('x_o=', X_o[0])

#print(x-X_o[:,0])
for i in range(len(t_eval)):
    print (x[i], X_o[i, 0]) 

fig = plt.figure()
ax = fig.add_subplot()
#ax1.plot(x, t, 'b-')
ax.plot(t, x - X_o[:, 0], 'g-', label='x - x_o')
ax.set_xlabel('t [с]')
ax.set_ylabel('X [м]')


plt.show()
