import deepxde as dde
from deepxde.backend import tf

import matplotlib.pyplot as plt
import numpy as np

# ===================================================
# PDEs OF MODELS
# ===================================================
def pde(system_name, x, y, params):
  if system_name == "Newton's Law of Cooling":
    T_init, T_surr, k_cool = params
    dT_t = dde.grad.jacobian(y, x) 
    return dT_t + k_cool * (y - T_surr)
  elif system_name == "Simple Harmonic Oscillator":
    pass

# 1D geometry object of interval [-1, 1]
# Corresponds to x_train in prev. examples
geom = dde.geometry.Interval(0, 10)

# Defines boundary condition
# Returns True if x is on boundary
def boundary(x, on_boundary):
  return on_boundary

# Defines value of solution at boundary
# Represents Dirichlet, enforcing solution ot take 0 at boundary
def boundary_func(x):
  return 0

def exact_sol(x):
  return np.sin(np.pi * x)

# Create B.C.
bc = dde.icbc.DirichletBC(geom, boundary_func, boundary)
# Create data object that defines problem setup for training
# 16: number of training points sampled from domain
# 2: number of B.C.s
data = dde.data.PDE(geom, pde, bc, 16, 2, solution=exact_sol, num_test=100)

# NN Architecture
def architecture(neurons, layers):
  layer_size = [1] + [neurons] * layers + [1] # 1 input, 3 hidden, 1 output -> [1, 50, 50, 50, 1]
  activation = "tanh"
  initializer = "Glorot uniform" # Weight initialization
  net = dde.nn.FNN(layer_size, activation, initializer)
  return locals()


# Compile model
#model = dde.Model(data, net)
#model.compile("adam", lr=0.001, metrics=["l2 relative error"])

# Train model
#losshistory, train_state = model.train(iterations=1000)


# ==================== PLOT ====================
"""
x = geom.uniform_points(30, True)
y_pred = model.predict(x)
y_true = exact_sol(x)
plt.figure()
plt.plot(x, y_pred, label="PINN-Predicted")
plt.plot(x, y_true, '*',label="Analytical Soln.")
plt.xlabel('x')
plt.ylabel('PDE residual')
plt.legend(loc='best')
plt.show()
"""