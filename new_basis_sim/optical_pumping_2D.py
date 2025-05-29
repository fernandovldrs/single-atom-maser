import qutip
import matplotlib.pyplot as plt
import numpy as np
import time
from helpers import transmon, calculate_geff, calc_average_transmon, gaussian_ramp_envelope
from multiprocessing import Pool
import time
import itertools

###########################################################################
##                                                                       ##
## This script simulates the dynamics of the qubit as a function of      ##
## time under parametric modulation.                                     ##
## I'm doing this simulation in the charge basis, which considers        ##
## Non-adiabatic transitions and changes to driving parameters.          ##
##                                                                       ##
###########################################################################

# Load flux and transmon parameters from g_eff calculation
data = np.load("new_basis_sim\\sideband_geff_solution.npz", allow_pickle=True)
transmon_params = data["transmon_params"].item()
flux_params = data["flux_params"].item()


# # Load g_eff solutions for each set of flux modulation
# n_contours = max(int(k.split('_')[-1]) for k in data.files if k.startswith("As_")) + 1
# g_scaling_arr = np.concatenate([data[f"g_scaling_{i}"] for i in range(n_contours)])
# As_arr = np.concatenate([data[f"As_{i}"] for i in range(n_contours)])
# g_scaling_list = [tuple(g_scaling_arr[i:i+3]) for i in range(0, len(g_scaling_arr), 3)]
# As_list = [tuple(As_arr[i:i+2]) for i in range(0, len(As_arr), 2)]


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
g_scaling_list = [tuple(g_scaling_list[i:i+3]) for i in range(0, len(g_scaling_list), 3)][:30:-1]
As_list = [tuple(As_list[i:i+2]) for i in range(0, len(As_list), 2)][::-1]
# print(g_scaling_list[:30])
# print(g_scaling_list)

drive_params = {
    "A": 0.015, # GHz
    "freq": 0, #f_avg-alpha/2, # This is changed later, GHz 
}

drive_pulse_params = {
    "t0": 0,
    "pulse_len": 5000,
    "ramp_std": 10,
    "ramp_chop": 2,
}

rr_params = {
    "freq": 0, # (f_avg-alpha) + 2*flux_params["freqs"][0], # This is changed later, GHz
    "trunc": 2,
    "g": 0.030, # coupling, GHz
    "kappa": 1/30, # GHz
}


def run_simulation(sweep_param):

    drive_A, g_indx = sweep_param
    # Sweep over solutions
    # print(g_scaling_list[g_indx])
    g2, g0, gm2 = g_scaling_list[g_indx]
    A_flux1, A_flux2 = As_list[g_indx]

    # Prepare sweep parameters variables
    flux_params_new = flux_params.copy()
    flux_params_new["As"] = (A_flux1, A_flux2)

    f_avg, alpha, lambda01, lambda12 = calc_average_transmon(transmon_params, flux_params_new)
    
    drive_params_new = drive_params.copy()
    drive_params_new["A"] = drive_A
    drive_params_new["freq"] = f_avg-alpha/2
    
    rr_params_new = rr_params.copy()
    rr_params_new["freq"] = (f_avg-alpha) + 2*flux_params_new["freqs"][0]

    # Preliminary calculations for drive Hamiltonian
    s01 = qutip.basis(3, 0)*qutip.basis(3, 1).dag()
    s12 = qutip.basis(3, 1)*qutip.basis(3, 2).dag()
    delta_01 = 2*np.pi*(drive_params_new["freq"] - f_avg)
    delta_12 = 2*np.pi*(drive_params_new["freq"] - (f_avg - alpha))
    g_scaling = g0
    drive_env = gaussian_ramp_envelope(**drive_pulse_params)

    def H_drive(t, *args):

        drive_op = g_scaling*(np.exp(1j*delta_01*t)*lambda01*s01 + np.exp(1j*delta_12*t)*np.sqrt(2)*lambda12*s12)
        H = 2*np.pi*drive_env(t)*drive_params_new["A"]*drive_op

        return H + H.dag()

    # Preliminary calculations for coupling Hamiltonian
    a = qutip.destroy(rr_params_new["trunc"])
    delta_01_rr = 2*np.pi*(rr_params_new["freq"] - f_avg)
    delta_12_rr = 2*np.pi*(rr_params_new["freq"] - (f_avg - alpha)) 
    g_scaling = g2

    def H_coupling(t, *args):
        # pulse_params_new = pulse_params.copy() # Copy dictionary to set drive_len_value
        # pulse_params_new["pulse_len"] = args[0]["sweep_param"]
        coupling_op = g_scaling*(np.exp(1j*delta_01_rr*t)*lambda01*s01 + np.exp(1j*delta_12_rr*t)*np.sqrt(2)*lambda12*s12)
        coupling_op = qutip.tensor(coupling_op, a.dag())
        H = 2*np.pi*rr_params_new["g"]*coupling_op*np.exp(-1j*2*2*np.pi*min(flux_params_new["freqs"])*t)
        
        return H + H.dag()

    def H_total(t, *args):
        return qutip.tensor(H_drive(t, *args), qutip.qeye(rr_params_new["trunc"])) + H_coupling(t, *args)

    initial_state = qutip.tensor(qutip.basis(3, 0), qutip.basis(rr_params_new["trunc"], 0))
    t_list = np.arange(0, 5000, 1)
    c_ops = [np.sqrt(rr_params_new["kappa"])*qutip.tensor(qutip.qeye(3), qutip.destroy(rr_params_new["trunc"]))]

    start_time = time.time()  # Start timer
    result = qutip.mesolve(H_total, initial_state, t_list, c_ops = c_ops, args = {})
    print(f"Elapsed time: {time.time() - start_time:.6f} seconds")

    pop1 = [np.real((qutip.ket2dm(qutip.basis(3, 1))*state.ptrace(0)).tr()) for state in result.states]
    return pop1[-1]
    # return result.states

if __name__ == "__main__":

    drive_A_list = np.linspace(0.001, 0.080, 40)
    g_index_list = range(len(g_scaling_list))

    param_grid = list(itertools.product(drive_A_list, g_index_list))
    pool = Pool(processes=12, maxtasksperchild=1)  # Adjust the number of processes based on your CPU
    results = pool.map(run_simulation, param_grid)
    pool.close()
    pool.join()

    pop1_grid = np.array(results).reshape(len(drive_A_list), len(g_index_list))

    # Plot the 2D map
    plt.figure(figsize=(6, 5))
    extent = [g_index_list[0], g_index_list[-1], drive_A_list[0], drive_A_list[-1]]
    plt.imshow(pop1_grid, origin='lower', extent=extent, aspect='auto', cmap='viridis')
    plt.colorbar(label='Final |1⟩ population')
    plt.xlabel('g_res (GHz)')
    plt.ylabel('omega_drive (GHz)')
    plt.title('Population of |1⟩ vs omega_drive and g_res')
    plt.tight_layout()
    plt.show()
