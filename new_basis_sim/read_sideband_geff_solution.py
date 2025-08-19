import qutip
import matplotlib.pyplot as plt
import numpy as np
import time
from helpers import transmon, calculate_geff, calc_average_transmon, gaussian_ramp_envelope
from multiprocessing import Pool
import time
import itertools

# Load flux and transmon parameters from g_eff calculation
data = np.load("new_basis_sim\\sideband_geff_solution.npz", allow_pickle=True)
transmon_params = data["transmon_params"].item()
flux_params = data["flux_params"].item()

solution_indx = 15

# Collect g_scaling for every curve f_avg = f_target
contour_number = max(int(key.split('_')[-1]) 
                     for key in data.files 
                     if "As_" in key)
g_scaling_lists = []
As_lists = []
for contour_indx in range(contour_number + 1):
    g_scaling_lists.append(data[f"g_scaling_{contour_indx}"])
    As_lists.append(data[f"As_{contour_indx}"])

g_scaling_list = np.array(g_scaling_lists).reshape(-1)
As_list = np.array(As_lists).reshape(-1)
g_scaling_list = [tuple(g_scaling_list[i:i+3]) for i in range(0, len(g_scaling_list), 3)][::-1]
As_list = [tuple(As_list[i:i+2]) for i in range(0, len(As_list), 2)][::-1]

print(As_list[solution_indx])
print(g_scaling_list[solution_indx])