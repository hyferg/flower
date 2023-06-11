import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
fps = 60
animation_seconds = 3
num_points = 10000

loops = 30
num_petals = 5
petal_rotation = 1/4 * np.pi

# Generate the stem curve
squish = 4
gammas = np.linspace(0, 2 * np.pi / squish, num_points)
x = gammas
y = 0.25 * np.sin(gammas * squish)
stem_theta = 1 / 2 * np.pi
x_data_1 = x * np.cos(stem_theta) - y * np.sin(stem_theta)
y_data_1 = x * np.sin(stem_theta) + y * np.cos(stem_theta) - 2 * np.pi / squish

# Generating loops of angles from 0 to 2 pi
thetas = np.linspace(0, loops * 2 * np.pi - 1e-10, num_points)

# Generating amplitudes for each curve
amp_petals = np.vectorize(lambda theta: (np.cos(num_petals * (theta + petal_rotation)) + 1) / 2)(thetas)
amp_seeds = np.vectorize(lambda theta: 0.25)(thetas)

# Concentric levels for each loop beyond 2 pi
levels = (1 + np.floor_divide(thetas, 2 * np.pi)) / loops

# Generating complex numbers on the unit circle for each curve
z_petals = 0.7 * levels * amp_petals * np.exp(1j * thetas)
z_seeds = 0.7 * levels * amp_seeds * np.exp(1j * thetas)

# Extracting real and imaginary parts for each curve
x_petals = np.real(z_petals)
y_petals = np.imag(z_petals)

x_seeds = np.real(z_seeds)
y_seeds = np.imag(z_seeds)

# Delete some points for better plotting
diff_steps = np.insert(np.diff(levels), 0, 0)

x_petals[diff_steps > 0] = np.nan
y_petals[diff_steps > 0] = np.nan

x_seeds[diff_steps > 0] = np.nan
y_seeds[diff_steps > 0] = np.nan

# Create an empty figure and axes
fig, ax = plt.subplots()
scale = 2
ax.set_xlim(-scale, scale)
ax.set_ylim(-scale, scale)

# Set background color
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

# Create empty curves
curve_1, = ax.plot([], [], '#37C71E', linewidth=5)
curve_2, = ax.plot([], [], '#FAF9F6', linewidth=4)
curve_3, = ax.plot([], [], '#F7A014', linewidth=4)

# Initialize the curves to an empty state
def init():
    curve_1.set_data([], [])
    curve_2.set_data([], [])
    curve_3.set_data([], [])
    return curve_1, curve_2, curve_3

total_points = len(x_data_1) + len(x_petals) + len(x_seeds)

points_per_frame = total_points // (animation_seconds * fps)

animation_frames = int(np.floor(total_points / points_per_frame))

print(animation_frames)

def combined_update(i):
    # print(i, animation_frames, i * 2 * points_per_frame, i * 2 * points_per_frame >= total_points)
    # Calculate the index for plotting/unplotting
    i *= 2 * points_per_frame

    if i >= total_points:
        # We are in the second half of the animation, so unplot the points
        i -= total_points
        end_points = total_points - i

        if end_points > len(x_data_1):
            curve_2.set_data(x_petals[:end_points - len(x_data_1)], y_petals[:end_points - len(x_data_1)])
            curve_3.set_data(x_seeds[:end_points - len(x_data_1)], y_seeds[:end_points - len(x_data_1)])
        else:
            curve_2.set_data([], [])
            curve_3.set_data([], [])
            curve_1.set_data(x_data_1[:end_points], y_data_1[:end_points])
    else:
        # We are in the first half of the animation, so plot the points
        if i < len(x_data_1):
            curve_1.set_data(x_data_1[:i], y_data_1[:i])
        else:
            rem_i = i - len(x_data_1)
            curve_2.set_data(x_petals[:rem_i], y_petals[:rem_i])
            curve_3.set_data(x_seeds[:rem_i], y_seeds[:rem_i])

    return curve_1, curve_2, curve_3

# Create the animation
ani = FuncAnimation(fig, combined_update, frames=animation_frames, init_func=init, blit=True, interval=1000/fps)

# Show the animation
plt.show()
