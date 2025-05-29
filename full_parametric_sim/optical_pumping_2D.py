import qutip
import matplotlib.pyplot as plt
import numpy as np
import time
from helpers import transmon_charge
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

# Define frequency curve parameters
f_avg = 7.2  # in GHz
alpha = 0.2

# Define drive properties
N = 7 # Charge operator cutoff
# n = qutip.Qobj(np.diag(np.arange(-N, N+1))) # Charge operator
# omega_drive = 0.015*2*np.pi
freq_drive = - alpha/2 # At the GF/2 point, qubit rotation frame
phi_drive = 0
drive_ramp_std = 5
drive_t0 = 20

# Define readout resonator
rr_trunc = 3
# g_res = 0.015 # Coupling factor in GHz
kappa = 1/25 # Decay

# Define qubit
ref_transmon = transmon_charge(f_max = f_avg, alpha = -alpha, N = N)

# Find the change-of-basis matrix to the reference flux point and define post-COB dimension cutoff
transmon_trunc = 4 # Reduce from 2*N+1 dimensions to transmon_trunc
H_tr = ref_transmon.H_tr
cob_matrix = H_tr.eigenstates()[1]
transmon_Es = H_tr.eigenenergies()
H_offset =  transmon_Es[0]

# Update frequencies given the actual transmon transitions
freq_drive = ((transmon_Es[2] - transmon_Es[0])-2*(transmon_Es[1] - transmon_Es[0]) )/2/2/np.pi
rr_freq = ((transmon_Es[2] - transmon_Es[0])-2*(transmon_Es[1] - transmon_Es[0]) )/2/np.pi

H_tr_diag = H_tr.transform(cob_matrix)
H_tr_diag_offset = H_tr_diag-H_offset
H_transmon = qutip.Qobj(H_tr_diag_offset.tidyup(atol=1e-6)[:transmon_trunc,:transmon_trunc])

meas_basis = [qutip.basis(transmon_trunc, n) for n in range(transmon_trunc)]
proj_list = [qutip.ket2dm(state) for state in meas_basis[:3]] # Projector operators onto g, e and f

n_ch = qutip.Qobj(np.diag(np.arange(-N,N+1))) # charge operator
n_full = n_ch.transform(cob_matrix) # charge operator in eigenbasis
n = n_full[:transmon_trunc,:transmon_trunc]
n_r = np.copy(n) # ladder operator with upper triangule only
n_l = np.copy(n) # ladder operator with lower triangule only
for i in range(transmon_trunc):
    for j in range(transmon_trunc):
        if i>j:
            n_r[i][j] = 0
            n_l[j][i] = 0
n = qutip.Qobj(np.where(np.abs(n) < 1e-6, 0, n))
n_l = qutip.Qobj(np.where(np.abs(n_l) < 1e-6, 0, n_l))
n_r = qutip.Qobj(np.where(np.abs(n_r) < 1e-6, 0, n_r))

# Change to the rotating frame of the qubit
H_rot = qutip.Qobj(np.diag(np.arange(transmon_trunc)))*(transmon_Es[1] - transmon_Es[0])
H_transmon -= H_rot

# Sweep values
omega_drive_list = np.linspace(0.005, 0.030, 5)  # in 2π GHz
g_res_list = np.linspace(0.002, 0.020, 5)  # in GHz

def H_drive(t, *args):
    drive_len = args[0]["drive_len"]

    A = drive_t0
    B = drive_ramp_std
    C = drive_len

    if A < t < 2*B + A:
        # V = omega_drive*np.cos(2*np.pi*freq_drive*t + phi_drive)
        Vl = omega_drive*np.exp(-1j*2*np.pi*freq_drive*t + phi_drive)
        Vr = omega_drive*np.exp(1j*2*np.pi*freq_drive*t + phi_drive)
        Vl *= np.exp(-(t-(2*B + A))**2/2/B**2)
        Vr *= np.exp(-(t-(2*B + A))**2/2/B**2)
    elif 2*B + A <= t <= C + 2*B + A:
        Vl = omega_drive*np.exp(-1j*2*np.pi*freq_drive*t + phi_drive)
        Vr = omega_drive*np.exp(1j*2*np.pi*freq_drive*t + phi_drive)
    elif C + 2*B + A <= t <= C + 4*B + A:
        Vl = omega_drive*np.exp(-1j*2*np.pi*freq_drive*t + phi_drive)
        Vr = omega_drive*np.exp(1j*2*np.pi*freq_drive*t + phi_drive)
        Vl *= np.exp(-(t-(C + 2*B + A))**2/2/B**2)
        Vr *= np.exp(-(t-(C + 2*B + A))**2/2/B**2)
    else:
        Vl = 0
        Vr = 0
    return Vr*n_r + Vl*n_l

def run_sim(params):
    omega_drive_val, g_res_val = params
    # Redefine these inside function to scope correctly
    global omega_drive, g_res
    omega_drive = omega_drive_val * 2 * np.pi
    g_res = g_res_val
    # Redefine H_resonator
    a = qutip.destroy(rr_trunc)
    H_resonator = 2*np.pi*g_res*(qutip.tensor(qutip.Qobj(n_r), a.dag()) + qutip.tensor(qutip.Qobj(n_l), a))
    H_resonator += 2*np.pi*rr_freq*(qutip.tensor(qutip.qeye(transmon_trunc), a.dag()*a))
    def H_total(t, *args):
        return qutip.tensor(H_transmon + H_drive(t, *args), qutip.qeye(rr_trunc)) + H_resonator
    initial_state = qutip.tensor(meas_basis[0], qutip.basis(rr_trunc, 0))
    t_list = np.arange(0, 2000, 1)
    c_ops = [np.sqrt(kappa)*qutip.tensor(qutip.qeye(transmon_trunc), qutip.destroy(rr_trunc))]
    args = {"drive_len": 3500}
    start_time = time.time()  # Start timer
    result = qutip.mesolve(H_total, initial_state, t_list, c_ops=c_ops, args=args)
    print(f"Elapsed time: {time.time() - start_time:.6f} seconds")
    pop1 = [np.real((proj_list[1]*state.ptrace(0)).tr()) for state in result.states]
    return pop1[-1]

if __name__ == "__main__":

    param_grid = list(itertools.product(omega_drive_list, g_res_list))
    # omega_drive_list = 2*np.pi*np.linspace(0.02, 0.10, 5)
    pool = Pool(processes=12, maxtasksperchild=1)  # Adjust the number of processes based on your CPU
    results = pool.map(run_sim, param_grid)
    pool.close()
    pool.join()

    # Convert results to 2D array
    pop1_grid = np.array(results).reshape(len(omega_drive_list), len(g_res_list))
    
    # Plot the 2D map
    plt.figure(figsize=(6, 5))
    extent = [g_res_list[0], g_res_list[-1], omega_drive_list[0], omega_drive_list[-1]]
    plt.imshow(pop1_grid, origin='lower', extent=extent, aspect='auto', cmap='viridis')
    plt.colorbar(label='Final |1⟩ population')
    plt.xlabel('g_res (GHz)')
    plt.ylabel('omega_drive (GHz)')
    plt.title('Population of |1⟩ vs omega_drive and g_res')
    plt.tight_layout()
    plt.show()