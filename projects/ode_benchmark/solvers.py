from scipy.integrate import solve_ivp

def RK4(ode_instance, y, t, h):
    k1 = ode_instance.derivs(y, t)
    k2 = ode_instance.derivs(y + 0.5 * k1 * h, t + 0.5 * h)
    k3 = ode_instance.derivs(y + 0.5 * k2 * h, t + 0.5 * h)
    k4 = ode_instance.derivs(y + k3 * h, t + h)

    return y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

def forward_euler(ode_instance, y, t, h):
  return y + h * ode_instance.derivs(y, t)

def heun_method(ode_instance, y, t, h):
  intval = y + h * ode_instance.derivs(y, t)
  return y + h/2 * (ode_instance.derivs(y, t) + ode_instance.derivs(intval, t+h))

def RK45(ode_instance, t_span, t_eval):
    sol = solve_ivp(
        lambda t, y: ode_instance.derivs(y, t), 
        t_span, 
        ode_instance.y0, 
        t_eval=t_eval,
        method='RK45',    
        rtol=1e-12,         
        atol=1e-12
        )
    return sol.y.T # Returns (steps, states)