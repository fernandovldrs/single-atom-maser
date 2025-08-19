import qutip
import matplotlib.pyplot as plt
import numpy as np
# import pickle
import os
from helpers import transmon, calculate_geff, calc_average_transmon, gaussian_ramp_envelope

###########################################################################
##                                                                       ##
## This script simulates the dynamics of the qubit as a function of     ##
## time under parametric modulation.                                    ##
## I'm doing this simulation in the charge basis, which considers       ##
## Non-adiabatic transitions and changes to driving parameters.         ##
##                                                                       ##
###########################################################################

transmon_trunc = 3

# Sweep over drive strength
drive_A_list = np.arange(0.002, 0.102, 0.002)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
labels = ['A', 'B', 'C']#['A', 'B', 'C', 'D']  # For sol = 2, 4, 0 respectively
# my_data = []

for label, sol in zip(labels, [2,4,0]):#[2, 4, 0, 5]
    x_axis = []
    Nss_list = []
    F_list = []

    for drive_A in drive_A_list:
        filename = f'data/laser_threshold_{sol:d}/state_{drive_A*1e3:.0f}MHz.npz'
        if not os.path.exists(filename):
            continue
        
        x_axis.append(1e3*drive_A)
        loaded_data = np.load(filename)
        dims = loaded_data['dims'].tolist()
        final_state = qutip.Qobj(loaded_data['data'], dims=dims)

        a = qutip.tensor(qutip.destroy(dims[0][0]), 
                         qutip.qeye(transmon_trunc), 
                         qutip.qeye(2))

        mean_n = (a.dag() * a * final_state).tr()
        mean_n_squared = ((a.dag() * a) ** 2 * final_state).tr()
        variance_n = mean_n_squared - mean_n**2
        fano_number = variance_n / mean_n if mean_n != 0 else 0

        Nss_list.append(np.real(mean_n))
        F_list.append(np.real(fano_number))

    ax[0].plot(x_axis, Nss_list, label=label)
    ax[1].plot(x_axis, F_list, label=label)
    # a = {'x': x_axis, 'y_1': Nss_list, 'y_2': F_list}
    # my_data.append(a)


# with open('POSTER_DATA_PLOT.pickle', 'wb') as handle:
#     pickle.dump(my_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print(len(x_axis))

# Set axis labels and formatting for first plot
ax[0].set_ylabel(r"$\langle n \rangle$")
ax[0].set_xlabel("Drive amplitude (GHz)")
ax[0].set_title("Photon number")
ax[0].legend()
ax[0].grid(True)

# Add y-axis ticks every 20
# y_max = max([max(lst) for lst in [Nss_list]]) if Nss_list else 100
yticks = [0, 20, 40, 60, 80]
ax[0].set_yticks(yticks)

# Set axis labels and formatting for second plot
ax[1].set_ylabel("Fano factor")
ax[1].set_xlabel("Drive amplitude (GHz)")
ax[1].set_title("Fano factor")
ax[1].set_yscale("log")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()

