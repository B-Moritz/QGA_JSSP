import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
radius = 1
num_frames = 360
interval = 20  # milliseconds

# Setup the figure and axis
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

# Draw the unit circle
circle = plt.Circle((0, 0), radius, color='lightgray', fill=False)
ax.add_artist(circle)

# Initialize the vector line
vector_line, = ax.plot([], [], marker='o', color='blue')

# Initialization function
def init():
    vector_line.set_data([], [])
    return vector_line,

# Update function
def update(frame):
    angle = np.deg2rad(frame)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    vector_line.set_data([0, x], [0, y])
    return vector_line,

# Create the animation
ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=interval)

plt.title('Rotating Vector in Unit Circle')
plt.grid(alpha=0.3)
plt.show()