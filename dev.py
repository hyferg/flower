import numpy as np
import matplotlib.pyplot as plt

# Constants
num_petals = 5
petal_rotation = 1 / 4 * np.pi
petal_fill_loops = 30

fps = 60
animation_seconds = 3
num_points = 10000

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


# name
def chart(x, y, theta, offset: (float, float), mirror=False):
    if mirror:
        x = -x
    x_out = 2 * (x * np.cos(theta) - y * np.sin(theta) + offset[0])
    y_out = 2 * (x * np.sin(theta) + y * np.cos(theta) + offset[1])
    return x_out, y_out


x_1_pre = np.linspace(0.2, 3, 100)[:70]
y_2_pre = (lambda x: x ** x - x)(x_1_pre)
x_1, y_2 = chart(x_1_pre, y_2_pre, np.pi / 2.3, (0, 0), True)

x_name = np.concatenate((x_1, [None]))
y_name = np.concatenate((y_2, [None]))

# Create an empty figure and axes
fig, ax = plt.subplots()
scale = 20
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
curve_4, = ax.plot([], [], '#F7A014', linewidth=4)

# curve_1.set_data(x_stem, y_stem)
# curve_2.set_data(x_petals, y_petals)
# curve_3.set_data(x_seeds, y_seeds)
curve_3.set_data(x_name, y_name)

# Show the animation
plt.show()
