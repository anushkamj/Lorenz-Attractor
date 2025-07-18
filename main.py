import matplotlib
matplotlib.use('TkAgg')  # GUI backend for macOS - must be at the top

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Lorenz system differential equations
def lorenz(t, state, sigma=10, beta=8/3, rho=28):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Time range and evaluation points (fewer = faster)
t_span = (0, 40)
t_eval = np.linspace(*t_span, 1000)  # reduce points for speed

# Initial condition
initial_state = [1.0, 1.0, 1.0]

# Solve the system using Runge-Kutta (RK45)
sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, method='RK45')
x, y, z = sol.y

# Create figure and 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
line, = ax.plot([], [], [], lw=0.5, color='purple')  # empty line to update

# Set plot limits
ax.set_xlim((min(x), max(x)))
ax.set_ylim((min(y), max(y)))
ax.set_zlim((min(z), max(z)))
ax.set_title("Lorenz Attractor - Animated")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Update function for animation
def update(i):
    line.set_data(x[:i], y[:i])
    line.set_3d_properties(z[:i])
    return line,

# Animate with frame skipping for speed
ani = FuncAnimation(fig, update, frames=range(0, len(x), 5), interval=45, blit=True)

# Show the animated plot
plt.show()
