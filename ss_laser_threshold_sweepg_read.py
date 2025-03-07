import qutip
import matplotlib.pyplot as plt
import numpy as np
import time
from helper_fns import *
import scipy.sparse as sp
import os

###########################################################################
##                                                                       ##
## This script finds the steady-state solution of the laser dynamics as  ##
## a function of the pump amplitude.                                     ##
## It matches with previous theory in the literature.                    ##
## The drive interaction and capacitive couplings are derived            ##
## from the charge basis of the transmon.                                ##
##                                                                       ##
###########################################################################

g_list = [
    (0.5535, 0.5653), (0.5557, 0.5630), (0.5620, 0.5562), (0.5719, 0.5451),
    (0.5848, 0.5302), (0.5999, 0.5119), (0.6163, 0.4910), (0.6332, 0.4683),
    (0.6499, 0.4445), (0.6615, 0.4271), (0.6657, 0.4207), (0.6801, 0.3979),
    (0.6927, 0.3769), (0.7033, 0.3586), (0.7045, 0.3564), (0.7115, 0.3439),
    (0.7174, 0.3333), (0.7198, 0.3291), (0.7209, 0.3271), (0.7219, 0.3253),
    (0.7215, 0.3260), (0.7205, 0.3276), (0.7166, 0.3336), (0.7161, 0.3343),
    (0.7102, 0.3423), (0.7069, 0.3465), (0.7013, 0.3529), (0.6954, 0.3589),
    (0.6895, 0.3644), (0.6825, 0.3700), (0.6742, 0.3755), (0.6684, 0.3788),
    (0.6543, 0.3847), (0.6531, 0.3851), (0.6366, 0.3887), (0.6275, 0.3895),
    (0.6185, 0.3895), (0.5987, 0.3875), (0.5890, 0.3857), (0.5768, 0.3826),
    (0.5527, 0.3746), (0.5281, 0.3645), (0.5259, 0.3635), (0.4964, 0.3492),
    (0.4641, 0.3316), (0.4287, 0.3107), (0.4161, 0.3029), (0.3904, 0.2865),
    (0.3491, 0.2590), (0.3051, 0.2286), (0.2585, 0.1952), (0.2097, 0.1594),
    (0.1590, 0.1215), (0.1068, 0.0820), (0.0537, 0.0413)
]

for indx in [0]:#np.arange(len(g_list)-1, len(g_list)-1 - 11, -1):
    folder_path = f"sol_{indx}"

    # Simulation parameters
    res_trunc_list = [35, 70]
    transmon_trunc = 3
    aux_trunc = 2
    more_space = False

    # System parameters
    fge = 6600
    alpha = -200
    faux = alpha + fge
    # wgf2 = (fge + faux)/2
    g_res = 11*g_list[indx][1]  # 10MHz
    g_aux = 30*g_list[indx][0] # 30MHz
    # omega_gf2 = 24*2  # 20MHz
    kappa_res = 0.01*3  # T1 = 100us
    kappa_aux = 3.33  # T1 = 300ns

    omega_gf2_list = np.arange(0, 70, 2)
    Nss_list = []
    F_list = []

    for omega_gf2 in omega_gf2_list:
        filename = folder_path + f'/state_{omega_gf2:.0f}.npz'

        # Try opening with less dimensions
        try: 
            res_trunc = res_trunc_list[0]
            dims = [res_trunc, transmon_trunc, aux_trunc]
            d_total = res_trunc*transmon_trunc*aux_trunc
            loaded_data = np.load(filename)
            final_state = qutip.Qobj(loaded_data['data'], dims=loaded_data['dims'].tolist())
            a = qutip.tensor(qutip.destroy(res_trunc), qutip.qeye(transmon_trunc), qutip.qeye(aux_trunc))
            mean_n = (a.dag()*a* final_state).tr()
            mean_n_squared =  ((a.dag()*a)**2 * final_state).tr()
            variance_n = mean_n_squared - mean_n**2
            fano_number = variance_n / mean_n

        except: 
            res_trunc = res_trunc_list[1]
            dims = [res_trunc, transmon_trunc, aux_trunc]
            d_total = res_trunc*transmon_trunc*aux_trunc
            loaded_data = np.load(filename)
            final_state = qutip.Qobj(loaded_data['data'], dims=loaded_data['dims'].tolist())
            a = qutip.tensor(qutip.destroy(res_trunc), qutip.qeye(transmon_trunc), qutip.qeye(aux_trunc))
            mean_n = (a.dag()*a* final_state).tr()
            mean_n_squared =  ((a.dag()*a)**2 * final_state).tr()
            variance_n = mean_n_squared - mean_n**2
            fano_number = variance_n / mean_n

        Nss_list.append(mean_n)
        F_list.append(fano_number)

    plt.plot(omega_gf2_list, Nss_list, label = f"gaux = {g_list[indx][1]:.4f} gres = {g_list[indx][0]:.4f}")

plt.legend()
plt.show()
