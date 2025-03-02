import qutip
import matplotlib.pyplot as plt
import numpy as np
import time
from helper_fns import *
import scipy.sparse as sp

###########################################################################
##                                                                       ##
## This script finds the steady-state solution of the laser dynamics as  ##
## a function of the pump amplitude.                                     ##
## It matches with previous theory in the literature.                    ##
## The drive interaction and capacitive couplings are derived            ##
## from the charge basis of the transmon.                                ##
##                                                                       ##
###########################################################################

# Simulation parameters
res_trunc = 60
transmon_trunc = 3
aux_trunc = 2
dims = [res_trunc, transmon_trunc, aux_trunc]
d_total = res_trunc*transmon_trunc*aux_trunc

# System parameters
fge = 6000
alpha = -200
faux = alpha + fge
# wgf2 = (fge + faux)/2
g_res = 8 # 10MHz
g_aux = 15  # 30MHz
omega_gf2 = 24  # 20MHz
kappa_res = 0.2  # T1 = 100us
kappa_aux = 138  # T1 = 300ns

omega_gf2_list = np.arange(0, 50, 2)
Nss_list = []
F_list = []
for omega_gf2 in omega_gf2_list:

    # Define Hamiltonian and losses
    trs =  transmon(f_ge = fge, alpha = alpha, g_ef = g_aux, g_ge = g_res, 
                        gamma_res = kappa_res, kappa = kappa_aux, n_ph = res_trunc, f_q = omega_gf2, 
                        n_trunc = transmon_trunc, gamma_tr = 0)
    H = trs.build_H()
    c_ops = trs.build_C()
    H = qutip.Qobj(sp.csr_matrix(H.full(), dtype=complex))

    # Simulation
    # final_state = qutip.steadystate(H, c_ops, method = 'power', use_rcm = True)
    start_time = time.time()
    final_state = qutip.steadystate(H, c_ops, method = 'iterative')
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.6f} seconds")

    array = final_state.full()
    reshaped_array = array.reshape((res_trunc, 3, 2, res_trunc, 3, 2))
    reshaped_array = reshaped_array.reshape((d_total, d_total))

    # Convert back to Qobj
    final_state = qutip.Qobj(reshaped_array, dims=[[res_trunc, 3, 2], [res_trunc, 3, 2]])
    filename = f'state_{omega_gf2:.0f}'

    # Convert Qobj to NumPy array
    array = final_state.full()
    np.savez(filename, data=array, dims=final_state.dims)

## Read states

Nss_list = []
F_list = []

a = qutip.tensor(qutip.destroy(res_trunc), qutip.qeye(transmon_trunc), qutip.qeye(aux_trunc))

for omega_gf2 in omega_gf2_list:
    filename = f'state_{omega_gf2:.0f}.npz'
    loaded_data = np.load(filename)
    final_state = qutip.Qobj(loaded_data['data'], dims=loaded_data['dims'].tolist())

    mean_n = (a.dag()*a* final_state).tr()
    mean_n_squared =  ((a.dag()*a)**2 * final_state).tr()
    variance_n = mean_n_squared - mean_n**2
    fano_number = variance_n / mean_n
    print(mean_n, fano_number)

    Nss_list.append(mean_n)
    F_list.append(fano_number)

plt.plot(omega_gf2_list, Nss_list)
plt.show()
plt.plot(omega_gf2_list, F_list)
plt.show()