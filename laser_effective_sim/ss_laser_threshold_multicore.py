import qutip
import numpy as np
import time
from helpers import transmon
import scipy.sparse as sp
from multiprocessing import Pool
import os

import qutip.settings
# qutip.settings.num_cpus = 1

###########################################################################
##                                                                       ##
## This script finds the steady-state solution of the laser dynamics as  ##
## a function of the pump amplitude. It uses multicore processing.       ##
## It matches with previous theory in the literature.                    ##
## The drive interaction and capacitive couplings are derived            ##
## from the charge basis of the transmon.                                ##
##                                                                       ##
###########################################################################

def run_simulation(omega_gf2):

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
    g_indx = 45#len(g_list)-1-1-1-1-1
    folder_path = f"sol_{g_indx}_highQ" 
    os.makedirs(folder_path, exist_ok=True)
    
    # Simulation parameters
    res_trunc = 200
    transmon_trunc = 4
    aux_trunc = 2

    # System parameters
    fge = 6600
    alpha = -200
    faux = alpha + fge
    # wgf2 = (fge + faux)/2
    g_res = 11*g_list[g_indx][1]  # 10MHz
    g_aux = 30*g_list[g_indx][0] # 30MHz
    print(g_list[g_indx][1], g_list[g_indx][0] )
    # omega_gf2 = 24*2  # 20MHz
    kappa_res = 0.01  # T1 = 100us
    kappa_aux = 3.33  # T1 = 300ns
    gamma_tr = 0.05 # T1 = 20us

    dims = [res_trunc, transmon_trunc, aux_trunc]
    d_total = res_trunc*transmon_trunc*aux_trunc

    # Define Hamiltonian and losses
    trs =  transmon(f_ge = fge, alpha = alpha, g_ef = g_aux, g_ge = g_res, 
                        gamma_res = kappa_res, kappa = kappa_aux, n_ph = res_trunc, f_q = omega_gf2, 
                        n_trunc = transmon_trunc, gamma_tr = gamma_tr)
    H = trs.build_H()
    c_ops = trs.build_C()
    H = qutip.Qobj(sp.csr_matrix(H.full(), dtype=complex))
    
    # Simulation
    start_time = time.time()
    final_state = qutip.steadystate(H, c_ops, method = 'iterative')
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.6f} seconds")

    array = final_state.full()
    reshaped_array = array.reshape((res_trunc, transmon_trunc, 2, res_trunc, transmon_trunc, 2))
    reshaped_array = reshaped_array.reshape((d_total, d_total))

    # Convert back to Qobj
    final_state = qutip.Qobj(reshaped_array, dims=[[res_trunc, transmon_trunc, 2], [res_trunc, transmon_trunc, 2]])

    # Check whether more truncation is necessary
    proj = qutip.tensor(qutip.ket2dm(qutip.basis(res_trunc, res_trunc-1)),
                        qutip.qeye(transmon_trunc), qutip.qeye(2))
    pop = np.abs((final_state*proj).tr())
    if pop > 0.03:
        print(f"Consider increasing truncation (omega_gf2 = {omega_gf2})")

    filename = folder_path + f'/state_{omega_gf2:.0f}.npz'

    # Convert Qobj to NumPy array
    array = final_state.full()
    np.savez(filename, data=array, dims=final_state.dims)


if __name__ == "__main__":

    omega_gf2_list = [59]#np.linspace(1, 12*6-1, 12*3)[17:]

    pool = Pool(processes=1, maxtasksperchild=1)  # Adjust the number of processes based on your CPU
    results = pool.map(run_simulation, omega_gf2_list)
    pool.close()
    pool.join()
