import qutip
import matplotlib.pyplot as plt
import numpy as np
import time
from helpers import transmon_charge, gaussian_ramp_envelope
import scipy.sparse as sp
from multiprocessing import Pool
import time
from scipy.optimize import curve_fit

###########################################################################
##                                                                       ##
## This script simulates the dynamics of the qubit as a function of      ##
## time under parametric modulation.                                     ##
## I'm doing this simulation in the charge basis, which considers        ##
## Non-adiabatic transitions and changes to driving parameters.          ##
##                                                                       ##
###########################################################################

t_list = np.arange(0, 200*2, 0.1)

# Define transmon parameters
transmon_params = {
    "f0": 8.0, # GHz
    "d": 0.454, # SQUID asymmetry
    "alpha": 0.2, # - anharmonicity, GHz
    "N": 7, # Charge offset truncation
    "trunc": 9, # transmon Hamiltonian truncation
    "meas_flux": 0, # flux of measurement basis
}

# Define flux drive
p = 3
flux_mod_params = {
    "As": (0.332, 0.0),
    "freqs": (0.275, p*0.275), # GHz
    "phases": (0.0, 0.25), # rad/(2pi) 
}
flux_pulse_params = {
    "t0": 0,
    "pulse_len": 160*2,
    "ramp_std": 10,
    "ramp_chop": 2,
}

# Define charge drive
drive_mod_params = {
    "A": 0.001, # GHz
    "freq": 0, # GHz 
}
drive_pulse_params = {
    "t0": 20,
    "pulse_len": 276,
    "ramp_std": 5,
    "ramp_chop": 2,
}

# Transmon Hamiltonian generator
def transmon(flux, f0, alpha, d, N, **kwargs):
    return transmon_charge(f_max = f0, alpha = -alpha, d = d, flux = flux, N = N)

# Define qubit measurement at a reference flux point
transmon_trunc = transmon_params["trunc"] 
ref_transmon = transmon(transmon_params["meas_flux"], **transmon_params)
meas_basis = [qutip.basis(transmon_trunc, n) for n in range(transmon_trunc)]
proj_list = [qutip.ket2dm(state) for state in meas_basis] # Projector operators

# Find the change-of-basis matrix to the reference flux point and reduce dimension post-COB
cob_matrix = ref_transmon.H_tr.eigenstates()[1]

# Calculate charge operators
n_ch = qutip.Qobj(np.diag(np.arange(-transmon_params["N"], transmon_params["N"]+1))) # charge operator
n_full = n_ch.transform(cob_matrix) # charge operator in eigenbasis
n = n_full[:transmon_trunc,:transmon_trunc]
nr = np.copy(n) # ladder operator with upper triangule only
nl = np.copy(n) # ladder operator with lower triangule only
for i in range(transmon_trunc):
    for j in range(transmon_trunc):
        if i>j:
            nr[i][j] = 0
            nl[j][i] = 0
n = qutip.Qobj(np.where(np.abs(n) < 1e-6, 0, n))
nl = qutip.Qobj(np.where(np.abs(nl) < 1e-6, 0, nl))
nr = qutip.Qobj(np.where(np.abs(nr) < 1e-6, 0, nr))

# Offset Hamiltonian to be removed
H_offset =  ref_transmon.H_tr.eigenenergies()[0]

def flux_modulation(t, mod_params, pulse_params):
    flux_mod = sum([A*np.cos(2*np.pi*(freq*t + theta)) 
                     for A, freq, theta in zip(mod_params["As"], mod_params["freqs"], mod_params["phases"])])
    env = gaussian_ramp_envelope(**pulse_params)
    return  env(t) * flux_mod

def H_transmon(t, *args):

    # Find instantaneous flux point
    flux = flux_modulation(t, flux_mod_params, flux_pulse_params)
    H = transmon(flux, **transmon_params).H_tr

    # Change hamiltonian to reference basis
    H_tr_diag = H.transform(cob_matrix)
    H_tr_diag_offset = H_tr_diag-H_offset
    H = qutip.Qobj(H_tr_diag_offset.tidyup(atol=1e-6)[:transmon_trunc,:transmon_trunc])

    return H

def H_drive(t, mod_params, pulse_params, *args):

    mod_params_new = mod_params.copy() # Copy dictionary to set drive_len_value
    mod_params_new["freq"] = args[0]["sweep_param"]

    Vl = 2*np.pi*mod_params_new["A"]*np.exp(-1j*2*np.pi*mod_params_new["freq"]*t)
    Vr = np.conj(Vl)
    drive_mod = Vr*nr + Vl*nl
    env = gaussian_ramp_envelope(**pulse_params)

    return env(t)*drive_mod

def H_total(t, *args):

    freq_drive = args[0]["sweep_param"]

    # Change to the rotating frame of the drive
    f_rot = freq_drive
    H_rot = qutip.Qobj(np.diag(np.arange(transmon_trunc)))*2*np.pi*f_rot

    U_rot = (1j*H_rot*t).expm()
    H = H_transmon(t, *args) + H_drive(t, drive_mod_params, drive_pulse_params, *args) - H_rot

    return U_rot*(H)*U_rot.dag()

def run_simulation(sweep_param):

    initial_state = meas_basis[0]
    start_time = time.time()  # Start timer

    result = qutip.mesolve(H_total, initial_state, t_list, args = {"sweep_param": sweep_param})

    print(f"Elapsed time: {time.time() - start_time:.6f} seconds")

    return result.states

if __name__ == "__main__":

    # drive_freq_list = np.linspace(7.15175-0.03, 7.15175+0.03, 16*4)
    drive_freq_list = np.linspace(7.0475-0.015, 7.0475+0.015, 16*4)

    pool = Pool(processes=16, maxtasksperchild=1)  # Adjust the number of processes based on your CPU
    results = pool.map(run_simulation, drive_freq_list)
    pool.close()
    pool.join()

    states = np.array(results)
    
    # Create figure with gridspec for side-by-side layout
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 3])

    # Flux modulation plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_list, [flux_modulation(t, flux_mod_params, flux_pulse_params) for t in t_list])
    ax1.set_ylabel("Flux modulation")
    ax1.grid()

    # Drive envelope plot
    env = gaussian_ramp_envelope(**drive_pulse_params)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_list, [env(t) for t in t_list])
    ax2.set_ylabel("|Vr(t)|")
    ax2.set_xlabel("Time")
    ax2.grid()
        
    # Transmon populations
    ax3 = fig.add_subplot(gs[:, 1])  # spans both rows
    for level in range(3):
        ax3.plot(drive_freq_list, 
                 [np.real((proj_list[level]*qutip.ket2dm(s)).tr()) for s in states[:, -1]],
                 label=level)
    ax3.set_ylabel("Population")
    ax3.set_xlabel("Drive frequency")
    ax3.legend()
    ax3.grid()

    plt.tight_layout()
    plt.show()

