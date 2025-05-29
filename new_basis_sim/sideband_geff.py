import qutip
import matplotlib.pyplot as plt
import numpy as np
import time
from helpers import transmon, calculate_geff, calc_average_transmon, gaussian_ramp_envelope
from multiprocessing import Pool
import time

###########################################################################
##                                                                       ##
## This script simulates the dynamics of the qubit as a function of      ##
## time under parametric modulation.                                     ##
## I'm doing this simulation in the charge basis, which considers        ##
## Non-adiabatic transitions and changes to driving parameters.          ##
##                                                                       ##
###########################################################################


flux_params = {
    "As": (0.332, 0.0), # These parameters are swept
    "freqs": (0.275, 3*0.275), # GHz
    "phases": (0.0, 0.25), # rad/(2pi) 
}

transmon_params = {
    "fmax": 8.0, # GHz
    "d": 0.454, # SQUID asymmetry
    "alpha": 0.2, # - anharmonicity, GHz
} 

# f_avg, alpha, lambda01, lambda12 = calc_average_transmon(transmon_params, flux_params)

target_avg_f = 7.15
wm = flux_params["freqs"]

# Sweep parameters
A_flux_max = 0.35
A_flux1_vals = np.linspace(0, A_flux_max, 40)
A_flux2_vals = np.linspace(0, A_flux_max, 40)

# Compute avg_f for each combination of A_flux1 and A_flux2
avg_f_map = np.zeros((len(A_flux1_vals), len(A_flux2_vals)))

for i, A_flux1 in enumerate(A_flux1_vals):
    for j, A_flux2 in enumerate(A_flux2_vals):

        params_new = flux_params.copy() # Copy dictionary to set drive_len_value
        params_new["As"] = (A_flux1, A_flux2)
        
        avg_f_map[i, j] = calc_average_transmon(transmon_params, params_new)[0]

avg_f_map
# Add a contour line at avg_f
contour_levels = [target_avg_f]
fig, axes = plt.subplots(1, 2, figsize = (7, 4*0.8))
contour = axes[0].contour(A_flux1_vals, A_flux2_vals, avg_f_map, levels=contour_levels, colors='red', linewidths=2)

# Plot the results
axes[0].imshow(avg_f_map, extent=[0, A_flux_max, 0, A_flux_max], origin='lower', aspect='auto', cmap='viridis')
# axes[0].colorbar(label='avg_f (GHz)')

axes[0].set_xlabel('A_flux1')
axes[0].set_ylabel('A_flux2')
axes[0].set_title('Avg Frequency Response Map')

contour_points = []
for level_segs in contour.allsegs:
    for seg in level_segs:
        contour_points.append(seg)

# Print the extracted contour points
# Sideband scaling
g2_scaling = 30
gm2_scaling = 11
g_list = []

g_scaling_lists = []
for i, points in enumerate(contour_points):
    g_scaling_list = []
    
    for point in points:

        params_new = flux_params.copy() # Copy dictionary to set drive_len_value
        A_flux1, A_flux2 = point
        params_new["As"] = (A_flux1, A_flux2)

        g2, g0, gm2 = calculate_geff(transmon_params, params_new, Ns=[2, 0, -2], max_nk = 5)
        g_scaling_list.append((g2, g0, gm2))
        g_list.append((g2_scaling*g2, gm2_scaling*gm2))

    g_scaling_lists.append(g_scaling_list)

np.savez("new_basis_sim\\sideband_geff_solution.npz", 
         target_f_avg = target_avg_f,
         Ns = (2, 0, -2), 
         transmon_params = transmon_params,
         flux_params = flux_params,
         **{f"g_scaling_{i}": gs for i, gs in enumerate(g_scaling_lists)},
         **{f"As_{i}": points for i, points in enumerate(contour_points)}, 
         )
        
axes[1].scatter([g[0] for g in g_list], [g[1] for g in g_list])
axes[1].plot([g[0] for g in g_list], [g[1] for g in g_list])
axes[1].set_xlabel('geff N = 2')
axes[1].set_ylabel('geff N = -2')
axes[1].set_xlim([-0.1*g2_scaling, 0.8*g2_scaling])
axes[1].set_ylim([-0.1*gm2_scaling, 0.7*gm2_scaling])
axes[1].set_yticks([0, 3, 6])
axes[1].set_title(f'Coupling scaling at avg_f = {target_avg_f:.2f}GHz')
plt.grid()
plt.show()