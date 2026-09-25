from scipy.integrate import solve_ivp

class target():
    def __init__(self, pos, vel):

        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]

        self.vx = vel[0]
        self.vy = vel[1]
        self.vz = vel[2]

    def model(self, t, state):
        x, y, z, vx, vy, vz = state
        dxdt = vx
        dydt = vy
        dzdt = vz

        return [dxdt, dydt, dzdt, 0.0, 0.0, 0.0]

    def integrate(self, t_span):
        sol = solve_ivp(
            fun=self.model,
            t_span=t_span,
            y0=[self.x, self.y, self.z, self.vx, self.vy, self.vz],
            t_eval=None,
            r_tol=1e-6,
            a_tol=1e-8,
            method='RK45'
        )
        return sol
