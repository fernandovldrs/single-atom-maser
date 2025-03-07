import numpy as np
import matplotlib.pyplot as plt
from helper_fns import *
from flux_noise import calc_flux_noise
from calculate_geff import calculate_geff

# Define constants
f0 = 8  # in GHz
d = 0.454
p = 3
phi_dc = 0
w_flux_base = 2 * np.pi * 0.275
target_avg_f = 6.3

flux_theta_list = 2*np.pi*np.arange(0.30, 0.501, 0.005)
fig, axes = plt.subplots(1, 3)
optimal_points_list = []
for flux_theta in flux_theta_list:
    def flux_modulation(t, A_flux1, A_flux2, d):
        flux = A_flux1 * np.cos(w_flux_base * t) + A_flux2 * np.cos(w_flux_base * p * t + flux_theta) + phi_dc
        return f_scale(flux, d)

    # Sweep parameters
    A_flux1_min = 0.0
    A_flux2_min = 0.0
    A_flux1_max = 0.85
    A_flux2_max = 0.85

    A_flux1_vals = np.linspace(A_flux1_min, A_flux1_max, 50) # x axis
    A_flux2_vals = np.linspace(A_flux2_min, A_flux2_max, 50) # y axis

    t_list = np.arange(0, 150, 0.02)

    # Compute avg_f for each combination of A_flux1 and A_flux2
    avg_f_map = np.zeros((len(A_flux1_vals), len(A_flux2_vals)))

    for i, A_flux2 in enumerate(A_flux1_vals):
        for j, A_flux1 in enumerate(A_flux2_vals):
            avg_f_map[i, j] = np.mean([flux_modulation(t, A_flux1, A_flux2, d) * f0 for t in t_list])

    # Add a contour line at avg_f
    contour_levels = [target_avg_f]
    contour = axes[0].contour(A_flux1_vals, A_flux2_vals, avg_f_map, levels=contour_levels, colors='red', linewidths=2)

    # Extract contour points
    contour_points = []
    for collection in contour.collections:
        for path in collection.get_paths():
            contour_points.append(path.vertices)  # Store the x, y coordinates of the contour

    # Print the extracted contour points
    flux_noise_list = []
    for i, points in enumerate(contour_points):
        # Calculate flux noise
        # axes[0].scatter([point[0] for point in points], [point[1] for point in points])
        for point in points:
            A_flux1, A_flux2 = point
            print(np.mean([flux_modulation(t, A_flux1, A_flux2, d) * f0 for t in t_list]))
            flux_noise_list.append(calc_flux_noise(f0, d, p, phi_dc, A_flux1, A_flux2, flux_theta)/1e6/1e-9)

    try: # if there are solutions
        middle_indx = int(len(flux_noise_list)/2)
        flux_noise_list_A = flux_noise_list[:middle_indx]
        flux_noise_list_B = flux_noise_list[middle_indx:]
        flux_noise_A_indx = np.argmin(flux_noise_list_A)
        flux_noise_B_indx = np.argmin(flux_noise_list_B) + middle_indx
        point_A = points[flux_noise_A_indx]
        point_B = points[flux_noise_B_indx]
        optimal_points = [points[flux_noise_A_indx], 
                        points[flux_noise_B_indx]]
        optimal_points_list.append(optimal_points)
        axes[1].scatter([flux_noise_A_indx, flux_noise_B_indx], 
                        [flux_noise_list[flux_noise_A_indx], flux_noise_list[flux_noise_B_indx]])
    except:
        print(f"No solution found for phase {flux_theta/2/np.pi:.3f}*2pi")

    # Plot flux noise curve
    axes[1].plot(range(len(flux_noise_list)), flux_noise_list)
    
    # Plot coupling factors
    g0_A = calculate_geff(point_A[0], point_A[1], f0, d, p, w_flux_base, N= 0, phase = flux_theta)
    g2_A = calculate_geff(point_A[0], point_A[1], f0, d, p, w_flux_base, N= 2, phase = flux_theta)
    g0_B = calculate_geff(point_B[0], point_B[1], f0, d, p, w_flux_base, N= 0, phase = flux_theta)
    g2_B = calculate_geff(point_B[0], point_B[1], f0, d, p, w_flux_base, N= 2, phase = flux_theta)
    axes[2].scatter([g0_A, g0_B], [g2_A, g2_B])

    # Plot optimal points in the average frequency plot
    axes[0].scatter([point_A[0], point_B[0]], [point_A[1], point_B[1]])


# axes[1].set_xlabel('geff N = 2')
# axes[1].set_ylabel('geff N = -2')
# axes[1].set_xlim([-0.5, 1.1])
# axes[1].set_ylim([-0.5, 1.1])

axes[0].set_ylabel('A_flux2')
axes[0].set_xlabel('A_flux1')
axes[0].set_title('Avg Frequency Response Map')

axes[1].set_title(f'Relative flux noise at avg_f = {target_avg_f:.2f}GHz')

axes[2].set_ylabel('g N = 2')
axes[2].set_xlabel('g N = 0')
axes[2].set_title(f'Coupling scaling at sweet spots')
plt.grid()
plt.show()
    