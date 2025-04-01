import qutip
import matplotlib.pyplot as plt
import numpy as np
import time
from helper_fns import *
import scipy.sparse as sp
from multiprocessing import Pool
import time

###########################################################################
##                                                                       ##
## This script simulates the dynamics of the qubit as a function of      ##
## time under parametric modulation.                                     ##
## I'm doing this simulation in the charge basis, which considers        ##
## Non-adiabatic transitions and changes to driving parameters.          ##
##                                                                       ##
###########################################################################

# Define frequency curve parameters
f0 = 8  # in GHz
d = 0.454
alpha = 0.2

# Define flux modulation parameters
p = 3
# w_flux_base = 2 * np.pi * 0.275
flux_theta = 0.25*2*np.pi
A_flux1 = 0.332
A_flux2 = 0.0
flux_modulation_len = 180 + 250
flux_modulation_t0 = 0 
flux_modulation_ramp_std = 10

# Define Charge operator and drive
N = 7 # Operator cutoff
n = qutip.Qobj(np.diag(np.arange(-N, N+1))) # Charge operator
omega_drive = 0.00
freq_drive = 7.15
phi_drive = 0
drive_ramp_std = 5
drive_len = 110
drive_t0 = 20

# Define qubit measurement at a given flux point
flux_meas = 0
meas_basis = transmon_charge(f_max = f0, alpha = -alpha, d = d, flux = flux_meas, N = N).get_eigenbasis()
proj_list = [qutip.ket2dm(state) for state in meas_basis[:3]] # Projector operators onto g, e and f


def flux_modulation(t, A_flux1, A_flux2, w_flux_base):
    A = flux_modulation_t0
    B = flux_modulation_ramp_std
    C = flux_modulation_len

    if A < t < 3*B + A:
        flux = A_flux1 * np.cos(w_flux_base * t) + A_flux2 * np.cos(w_flux_base * p * t + flux_theta)
        flux *= np.exp(-(t-(3*B + A))**2/2/B**2)
        return flux
    elif 3*B + A <= t <= C + 3*B + A:
        return A_flux1 * np.cos(w_flux_base * t) + A_flux2 * np.cos(w_flux_base * p * t + flux_theta)
    elif C + 3*B + A <= t <= C + 6*B + A:
        flux = A_flux1 * np.cos(w_flux_base * t) + A_flux2 * np.cos(w_flux_base * p * t + flux_theta)
        flux *= np.exp(-(t-(C + 3*B + A))**2/2/B**2)
        return flux
    else:
        return 0


# plt.plot(np.arange(0, 300, 0.05), [flux_modulation(t, A_flux1, A_flux2) for t in np.arange(0, 300, 0.05)])
# plt.show()

def H_analog(t, *args):
    w_flux_base = args[0]["sweep_param"]
    # Find instantaneous flux point
    flux = flux_modulation(t, A_flux1, A_flux2, w_flux_base)
    H = transmon_charge(f_max = f0, alpha = -alpha, d = d, flux = flux, N = N).H_tr
    return H  

def H_drive_envelope(t, freq_drive):

    A = drive_t0
    B = drive_ramp_std
    C = drive_len

    if A < t < 2*B + A:
        V = omega_drive*np.cos(2*np.pi*freq_drive*t + phi_drive)
        V *= np.exp(-(t-(2*B + A))**2/2/B**2)
        return V
    elif 2*B + A <= t <= C + 2*B + A:
        V = omega_drive*np.cos(2*np.pi*freq_drive*t + phi_drive)
        return V
    elif C + 2*B + A <= t <= C + 4*B + A:
        V = omega_drive*np.cos(2*np.pi*freq_drive*t + phi_drive)
        V *= np.exp(-(t-(C + 2*B + A))**2/2/B**2)
        return V
    else:
        return 0

def H_drive(t, *args):
    # omega_drive = args[0]["omega_drive"]
    # drive_len = args[0]["drive_len"]
    # freq_drive = args[0]["sweep_param"]

    A = drive_t0
    B = drive_ramp_std
    C = drive_len

    if A < t < 2*B + A:
        V = omega_drive*np.cos(2*np.pi*freq_drive*t + phi_drive)
        V *= np.exp(-(t-(2*B + A))**2/2/B**2)
        return V*n
    elif 2*B + A <= t <= C + 2*B + A:
        V = omega_drive*np.cos(2*np.pi*freq_drive*t + phi_drive)
        return V*n
    elif C + 2*B + A <= t <= C + 4*B + A:
        V = omega_drive*np.cos(2*np.pi*freq_drive*t + phi_drive)
        V *= np.exp(-(t-(C + 2*B + A))**2/2/B**2)
        return V*n
    else:
        return 0

def H_total(t, *args):
    return H_analog(t, *args) + H_drive(t, *args)


def run_simulation(sweep_param):
    initial_state = meas_basis[0]
    t_list = np.arange(0, 250 + 250, 0.002)
    start_time = time.time()  # Start timer
    args = {"sweep_param" : sweep_param}
    result = qutip.mesolve(H_total, initial_state, t_list, args = args)
    final_state = result.states[-1]
    pop0 = (proj_list[0]*final_state*final_state.dag()).tr()
    pop1 = (proj_list[1]*final_state*final_state.dag()).tr()
    pop2 = (proj_list[2]*final_state*final_state.dag()).tr()
    print(f"Elapsed time: {time.time() - start_time:.6f} seconds")
    return pop0, pop1, pop2

if __name__ == "__main__":

    flux_modulation_list = 2*np.pi*np.linspace(50, 428*4, 16*16)/1000

    pool = Pool(processes=16, maxtasksperchild=1)  # Adjust the number of processes based on your CPU
    results = pool.map(run_simulation, flux_modulation_list)
    pool.close()
    pool.join()

    pop_list = np.array(results)

    plt.scatter(flux_modulation_list/2/np.pi, pop_list[:, 0])
    # plt.scatter(flux_modulation_list/2/np.pi, pop_list[:, 1])
    # plt.scatter(flux_modulation_list, pop_list[:, 2])
    plt.grid()

    plt.ylabel("Ground state population")
    plt.xlabel("Frequency")
    plt.show()

