import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
num_petals = 5
petal_rotation = 1 / 4 * np.pi
petal_fill_loops = 30

num_points = 10000
fps = 60
animation_seconds = 4

# Generate the stem curve
squish = 4
gammas = np.linspace(0, 2 * np.pi / squish, num_points)
x_pre = gammas
y_pre = 0.25 * np.sin(gammas * squish)
stem_rotation = 1 / 2 * np.pi
x_stem = x_pre * np.cos(stem_rotation) - y_pre * np.sin(stem_rotation)
y_stem = x_pre * np.sin(stem_rotation) + y_pre * np.cos(stem_rotation) - 2 * np.pi / squish

# Generating loops of angles from 0 to 2 pi
thetas = np.linspace(0, petal_fill_loops * 2 * np.pi - 1e-10, num_points)

# Generating amplitudes for each curve
amp_petals = np.vectorize(lambda theta: (np.cos(num_petals * (theta + petal_rotation)) + 1) / 2)(thetas)
amp_seeds = np.vectorize(lambda theta: 0.25)(thetas)

# Concentric levels for each loop beyond 2 pi
levels = (1 + np.floor_divide(thetas, 2 * np.pi)) / petal_fill_loops

# Generating complex numbers on the unit circle for each curve
z_petals = 0.7 * levels * amp_petals * np.exp(1j * thetas)
z_seeds = 0.7 * levels * amp_seeds * np.exp(1j * thetas)

# Extracting real and imaginary parts for each curve
x_petals = np.real(z_petals)
y_petals = np.imag(z_petals)

x_seeds = np.real(z_seeds)
y_seeds = np.imag(z_seeds)


def chart(x, y, theta, offset: (float, float), mirror=False, scale=1.0):
    if mirror:
        x = -x
    x_out = scale * (x * np.cos(theta) - y * np.sin(theta)) + offset[0]
    y_out = scale * (x * np.sin(theta) + y * np.cos(theta)) + offset[1]
    return x_out, y_out


segment_points = 100

# sara.jpg
x_1_pre = np.linspace(0.2, 3, segment_points)[:int(0.67 * segment_points)]
y_1_pre = (lambda x: x ** x - x)(x_1_pre)
x_1, y_1 = chart(x_1_pre, y_1_pre, np.pi / 2.3, (0.25, -0.1), True)

x_2_pre = np.linspace(0.15, 2, segment_points)
y_2_pre = (lambda x: x ** x)(x_2_pre)
x_2, y_2 = chart(x_2_pre, y_2_pre, np.pi / 2.3, (-0.9 * 0.8, -0.9 * 0.8), True, 0.8)

x_3_pre = np.linspace(0.15, 1.7, segment_points)
y_3_pre = (lambda x: x ** x)(x_3_pre)
x_3, y_3 = chart(x_3_pre, y_3_pre, np.pi / 2.35, (-3 * 0.8, -0.98 * 0.8), True, 0.8)

x_4_pre = np.linspace(-5, 4, segment_points)
y_4_pre = (lambda x: np.e ** x)(x_4_pre)
x_4, y_4 = chart(x_4_pre, y_4_pre, 0, (-5.0, -1.64), True, 0.1)

x_5_pre = np.linspace(0, 2.5, segment_points)[:int(0.9 * segment_points)]
y_5_pre = (lambda x: x ** x / 3)(x_5_pre)
x_5, y_5 = chart(x_5_pre, y_5_pre, np.pi / 2.1, (-6.9, -0.8), True, 1.2)

x_6_pre = np.linspace(0, 5, segment_points)
y_6_pre = 0 * x_6_pre
x_6, y_6 = chart(x_6_pre, y_6_pre, np.pi / 1.95, (-10, -1.2))

x_name = np.concatenate((x_1, [None], x_2, [None], x_3, [None], x_4, [None], x_5, [None], x_6))
y_name = np.concatenate((y_1, [None], y_2, [None], y_3, [None], y_4, [None], y_5, [None], y_6))

# Create an empty figure and axes
fig, ax = plt.subplots()
ax_scale = 15
ax.set_xlim(-ax_scale, ax_scale)
ax.set_ylim(-ax_scale, ax_scale)

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
# curve_1, = ax.plot([], [], '#37C71E', linewidth=5)
# curve_2, = ax.plot([], [], '#FAF9F6', linewidth=4)
curve_3, = ax.plot([], [], '#F7A014', linewidth=4)
# curve_4, = ax.plot([], [], '#F7A014', linewidth=4)

# curve_1.set_data(x_stem, y_stem)
# curve_2.set_data(x_petals, y_petals)
# curve_3.set_data(x_seeds, y_seeds)
curve_3.set_data(x_name, y_name)


def init():
    curve_3.set_data([], [])
    # curve_3.set_data(x_name, y_name)
    return curve_3,


def combined_update(i):
    pct_complete = i / animation_frames
    end_pt = int(pct_complete * len(x_name))
    curve_3.set_data(x_name[:end_pt], y_name[:end_pt])
    print(i)
    return curve_3,


animation_frames = int(animation_seconds * fps)

ani = FuncAnimation(fig, combined_update, frames=animation_frames, init_func=init, blit=True, interval=1000 / fps)

# Show the animation
plt.show()
