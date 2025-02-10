import numpy as np
import matplotlib.pyplot as plt
from calculate_geff import calculate_geff

# Define constants
f0 = 8  # in GHz
d = 0.454
p = 3
w_flux_base = 2 * np.pi * 0.275
flux_theta = 1.2*np.pi
target_avg_f = 7.15

# Define functions
def f_scale(flux, d):
    return np.sqrt(np.abs(np.cos(np.pi * flux) * np.sqrt(1 + d**2 * np.tan(np.pi * flux)**2)))

def flux_modulation(t, A_flux1, A_flux2, d):
    flux = A_flux1 * np.cos(w_flux_base * t) + A_flux2 * np.cos(w_flux_base * p * t + flux_theta)
    return f_scale(flux, d)

# Sweep parameters
A_flux_max = 0.35
A_flux1_vals = np.linspace(0, A_flux_max, 30)
A_flux2_vals = np.linspace(0, A_flux_max, 30)
t_list = np.arange(0, 150, 0.02)

# Compute avg_f for each combination of A_flux1 and A_flux2
avg_f_map = np.zeros((len(A_flux1_vals), len(A_flux2_vals)))

for i, A_flux1 in enumerate(A_flux1_vals):
    for j, A_flux2 in enumerate(A_flux2_vals):
        avg_f_map[i, j] = np.mean([flux_modulation(t, A_flux1, A_flux2, d) * f0 for t in t_list])

# # Add a contour line at avg_f
contour_levels = [target_avg_f]
fig, axes = plt.subplots(1, 2)
contour = axes[0].contour(A_flux1_vals, A_flux2_vals, avg_f_map, levels=contour_levels, colors='red', linewidths=2)

# Plot the results
axes[0].imshow(avg_f_map, extent=[0, A_flux_max, 0, A_flux_max], origin='lower', aspect='auto', cmap='viridis')
# axes[0].colorbar(label='avg_f (GHz)')

axes[0].set_xlabel('A_flux1')
axes[0].set_ylabel('A_flux2')
axes[0].set_title('Avg Frequency Response Map')

# Extract contour points
contour_points = []
for collection in contour.collections:
    for path in collection.get_paths():
        contour_points.append(path.vertices)  # Store the x, y coordinates of the contour

# Print the extracted contour points
g_eff_list_1 = []
g_eff_list_2 = []
for i, points in enumerate(contour_points):
    # print(f"Contour {i}:")
    # print(points)  # Each entry is an array of [A_flux1, A_flux2] points
    for point in points:
        A_flux1, A_flux2 = point
        g_eff_list_1.append(calculate_geff(A_flux1, A_flux2, f0, d, p, w_flux_base, N= 2, phase = flux_theta))
        g_eff_list_2.append(calculate_geff(A_flux1, A_flux2, f0, d, p, w_flux_base, N= -2, phase = flux_theta))

axes[1].scatter(g_eff_list_1, g_eff_list_2)
axes[1].plot(g_eff_list_1, g_eff_list_2)
axes[1].set_xlabel('geff N = 2')
axes[1].set_ylabel('geff N = -2')
axes[1].set_xlim([-0.5, 1.1])
axes[1].set_ylim([-0.5, 1.1])
axes[1].set_title(f'Coupling scaling at avg_f = {target_avg_f:.2f}GHz')
plt.grid()
plt.show()
    