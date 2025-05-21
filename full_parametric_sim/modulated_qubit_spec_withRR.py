import qutip
import matplotlib.pyplot as plt
import numpy as np
import time
from helpers import transmon_charge
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

t_list = np.arange(0, 200, 0.1)

# Define frequency curve parameters
f0 = 8  # in GHz
d = 0.454
alpha = 0.2

# Define flux modulation parameters
p = 3
w_flux_base = 2 * np.pi * 0.275
flux_theta = 0.25*2*np.pi
A_flux1 = 0.332
A_flux2 = 0.0
flux_modulation_len = 160
flux_modulation_t0 = 0 
flux_modulation_ramp_std = 10

# Define drive properties
N = 7 # Charge operator cutoff
# n = qutip.Qobj(np.diag(np.arange(-N, N+1))) # Charge operator
omega_drive = 0.025*2*np.pi
# freq_drive = 7.048405 # 7.151453 
phi_drive = 0
drive_ramp_std = 5
drive_t0 = 20
drive_len = 64*2

# Define readout resonator
rr_trunc = 2
rr_freq = 6.945357 + 2*0.275# frequency
g_res = 0.030*0 # Coupling factor in GHz
kappa = 1/200 # Decay

# Define qubit measurement at a reference flux point
flux_meas = 0
ref_transmon = transmon_charge(f_max = f0, alpha = -alpha, d = d, flux = flux_meas, N = N)

# Find the change-of-basis matrix to the reference flux point and define post-COB dimension cutoff
transmon_trunc = 9 # Reduce from 2*N+1 dimensions to transmon_trunc
cob_matrix = ref_transmon.H_tr.eigenstates()[1]
H_offset =  ref_transmon.H_tr.eigenenergies()[0]

meas_basis = [qutip.basis(transmon_trunc, n) for n in range(transmon_trunc)]
proj_list = [qutip.ket2dm(state) for state in meas_basis] # Projector operators onto g, e and f

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

def flux_modulation(t, A_flux1, A_flux2):
    A = flux_modulation_t0
    B = flux_modulation_ramp_std
    C = flux_modulation_len

    if A < t < 2*B + A:
        flux = A_flux1 * np.cos(w_flux_base * t) + A_flux2 * np.cos(w_flux_base * p * t + flux_theta)
        flux *= np.exp(-(t-(2*B + A))**2/2/B**2)
        return flux
    elif 2*B + A <= t <= C + 2*B + A:
        return A_flux1 * np.cos(w_flux_base * t) + A_flux2 * np.cos(w_flux_base * p * t + flux_theta)
    elif C + 2*B + A <= t <= C + 4*B + A:
        flux = A_flux1 * np.cos(w_flux_base * t) + A_flux2 * np.cos(w_flux_base * p * t + flux_theta)
        flux *= np.exp(-(t-(C + 2*B + A))**2/2/B**2)
        return flux
    else:
        return 0


def H_resonator(t, *args):
    # Resonator interaction picture
    a = qutip.destroy(rr_trunc)
    U_rot = (1j*2*np.pi*rr_freq*a.dag()*a*t).expm()
    a = U_rot*a*U_rot.dag()
    # return qutip.tensor(qutip.qeye(2*N+1), 2*np.pi*rr_freq*a.dag()*a) +  2*np.pi*g_res*qutip.tensor(n, a.dag() - a)
    return 2*np.pi*g_res*(  qutip.tensor(qutip.Qobj(n_r), a.dag()) + qutip.tensor(qutip.Qobj(n_l), a) ) 

def H_analog(t, *args):
    # Find instantaneous flux point
    flux = flux_modulation(t, A_flux1, A_flux2)
    H = transmon_charge(f_max = f0, alpha = -alpha, d = d, flux = flux, N = N).H_tr
    
    # Change hamiltonian to reference basis
    H_tr_diag = H.transform(cob_matrix)
    H_tr_diag_offset = H_tr_diag-H_offset
    H = qutip.Qobj(H_tr_diag_offset.tidyup(atol=1e-6)[:transmon_trunc,:transmon_trunc])

    return H

def H_drive(t, *args):
    freq_drive = args[0]["sweep_param"]
    
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

def drive_envelope(t, freq_drive):
    ## Just for plotting purposes

    A = drive_t0
    B = drive_ramp_std
    C = drive_len

    if A < t < 2*B + A:
        Vr = omega_drive*np.exp(1j*2*np.pi*freq_drive*t + phi_drive)
        Vr *= np.exp(-(t-(2*B + A))**2/2/B**2)
    elif 2*B + A <= t <= C + 2*B + A:
        Vr = omega_drive*np.exp(1j*2*np.pi*freq_drive*t + phi_drive)
    elif C + 2*B + A <= t <= C + 4*B + A:
        Vr = omega_drive*np.exp(1j*2*np.pi*freq_drive*t + phi_drive)
        Vr *= np.exp(-(t-(C + 2*B + A))**2/2/B**2)
    else:
        Vr = 0
    return np.abs(Vr)

def H_total(t, *args):
    freq_drive = args[0]["sweep_param"]
    # Change to the rotating frame of the drive
    f_rot = freq_drive
    H_rot = qutip.Qobj(np.diag(np.arange(transmon_trunc)))*2*np.pi*f_rot

    U_rot = qutip.tensor((1j*H_rot*t).expm(), qutip.qeye(rr_trunc))
    # n_rot = U_rot*n*U_rot.dag()
    H = qutip.tensor(H_analog(t, *args) + H_drive(t, *args) - H_rot, qutip.qeye(rr_trunc)) + H_resonator(t, *args)
    return U_rot*(H)*U_rot.dag()

def run_simulation(sweep_param):
    initial_state = qutip.tensor(meas_basis[0], qutip.basis(rr_trunc, 0))

    start_time = time.time()  # Start timer
    c_ops = []# [np.sqrt(kappa)*qutip.tensor(qutip.qeye(transmon_trunc), qutip.destroy(rr_trunc))]
    args = {"sweep_param": sweep_param}
    # result = qutip.mesolve(H_total, initial_state, t_list, c_ops = c_ops, args = args)
    result = qutip.mesolve(H_total, initial_state, t_list, args = args)
    final_state = result.states[-1].ptrace(0)
    pop_list = []
    for level in range(transmon_trunc):
        pop_list.append(np.real((proj_list[level]*final_state).tr()))
    print(f"Elapsed time: {time.time() - start_time:.6f} seconds")

    return np.array(pop_list)

if __name__ == "__main__":

    # drive_freq_list = np.linspace(7.151453-0.03, 7.151453+0.03, 16*4)
    drive_freq_list = np.linspace(7.045-0.05+0.1, 7.045+0.05+0.1, 16*6)

    pool = Pool(processes=16, maxtasksperchild=1)  # Adjust the number of processes based on your CPU
    results = pool.map(run_simulation, drive_freq_list)
    pool.close()
    pool.join()

    pop_list = np.array(results)
    
    # Create figure with gridspec for side-by-side layout
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 2])

    # Flux modulation plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_list, [flux_modulation(t, A_flux1, A_flux2) for t in t_list])
    ax1.set_ylabel("Flux modulation")
    ax1.grid()

    # |Vr(t)| plot
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_list, [drive_envelope(t, np.max(drive_freq_list)) for t in t_list])
    ax2.set_ylabel("|Vr(t)|")
    ax2.set_xlabel("Time")
    ax2.grid()

    # Power Rabi populations
    ax3 = fig.add_subplot(gs[:, 1])  # spans both rows
    ax3.plot(drive_freq_list, pop_list[:, 0], label='0')
    ax3.plot(drive_freq_list, pop_list[:, 1], label='1')
    ax3.plot(drive_freq_list, pop_list[:, 2], label='2')
    ax3.set_ylabel("Population")
    ax3.set_xlabel("Drive Length")
    ax3.legend()
    ax3.grid()

    plt.tight_layout()
    plt.show()

