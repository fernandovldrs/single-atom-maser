import numpy as np

# Load the file
data = np.load("new_basis_sim\\sideband_geff_solution.npz", allow_pickle=True)

# Load named arrays
target_f_avg = data["target_f_avg"]
Ns = tuple(data["Ns"])  # Convert from ndarray back to tuple

# Extract all g_scaling lists and contour points
# Determine how many there are based on the keys
g_scaling_lists = []
contour_points = []

contour_number = max(int(key.split('_')[-1]) 
                     for key in data.files 
                     if "As" in key)

# Iterate over all keys and categorize them
for contour_indx in range(contour_number + 1):
    g_scaling_lists.append(data[f"g_scaling_{contour_indx}"])
    contour_points.append(data[f"As_{contour_indx}"])

import matplotlib.pyplot as plt
plt.scatter(g_scaling_lists[0][:,0], g_scaling_lists[0][:,2])
i = 11
print(g_scaling_lists[0][i,0], g_scaling_lists[0][i,1], g_scaling_lists[0][i,2])
plt.scatter(g_scaling_lists[0][i,0], g_scaling_lists[0][i,2], c = 'r')
plt.show()
# Sanity check
print("target_f_avg:", target_f_avg)
print("Ns:", Ns)
print(f"{len(g_scaling_lists)} g_scaling arrays loaded.")
print(f"{len(contour_points)} contour point arrays loaded.")
