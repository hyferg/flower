import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fps = 24
animation_seconds = 2
num_points = 10000

loops = 30
petals = 5
rotation = 1/4 * np.pi

# Generating angles from 0 to 2 pi
thetas = np.linspace(0, loops * 2 * np.pi - 1e-10, num_points)

# Generating amplitudes
amp_1 = np.vectorize(lambda theta: ((np.cos(petals * (theta + rotation)) + 1)) / 2)(thetas)

# Concentric levels for each loop beyond 2 pi
levels = (1 + np.floor_divide(thetas, 2 * np.pi)) / loops

# Generating complex numbers on the unit circle
z = levels * amp_1 * np.exp(1j * thetas)

# Extracting real and imaginary parts
x_data = np.real(z)
y_data = np.imag(z)

# Delete some points for better plotting
diff_steps = np.insert(np.diff(levels), 0, 0)
x_data[diff_steps > 0] = np.nan
y_data[diff_steps > 0] = np.nan

# Create an empty figure and axes
fig, ax = plt.subplots()
scale = 2
ax.set_xlim(-scale, scale)
ax.set_ylim(-scale, scale)

# Set background color
# warm_off_white = (1, 0.98, 0.92)
background = "#000000"
fig.patch.set_facecolor(background)
ax.set_facecolor(background)

# Remove ticks, labels, and box
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_aspect('equal')
for spine in ax.spines.values():
    spine.set_visible(False)

# Create an empty curve
curve_1, = ax.plot([], [], '#FAF9F6', linewidth=4)

# Initialize the curve to an empty state
def init():
    curve_1.set_data([], [])
    return curve_1,

points_per_frame = len(x_data) // (animation_seconds * fps)

# Update function
def update(i):
    # Add the i-th point to the curve
    i *= points_per_frame
    curve_1.set_data(x_data[:i], y_data[:i])

    return curve_1,

# Create the animation
ani = FuncAnimation(fig, update, frames=int(np.floor(len(x_data) / points_per_frame)), init_func=init, blit=True, interval=1000/fps)

# Show the animation
plt.show()
