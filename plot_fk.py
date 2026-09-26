import matplotlib.pyplot as plt
import numpy as np
def plot_fk(ac, X_o, P_o, t_eval):

    labels = [
        'x [м]', 'y [м]', 'z [м]',
        'Vx [м/с]', 'Vy [м/с]', 'Vz [м/с]',
        'psi [рад]', 'theta [рад]', 'gamma [рад]',
        'omega_x [рад/с]', 'omega_y [рад/с]', 'omega_z [рад/с]'
    ]
    
    true = ac.sol.T       
    est  = X_o               
    err  = true - est
    
    sigma = np.sqrt(np.array([np.diag(P) for P in P_o])) 
    
    fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=True)
    axes = axes.ravel()
    
    for i in range(12):
        ax = axes[i]
        ax.plot(t_eval, err[:, i], 'b-', linewidth=1.0, label='невязка')
        ax.plot(t_eval,  3*sigma[:, i], 'r--', linewidth=0.8, label='+3СКО')
        ax.plot(t_eval, -3*sigma[:, i], 'r--', linewidth=0.8, label='-3СКО')
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.4)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel('t [с]')
    axes[-2].set_xlabel('t [с]')
    axes[-3].set_xlabel('t [с]')
    
    fig.suptitle('Невязки оценки состояния и трубки +-3СКО', fontsize=14)
    plt.tight_layout()
    plt.show()
    return fig