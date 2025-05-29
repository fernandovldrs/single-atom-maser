import qutip
import matplotlib.pyplot as plt
import numpy as np
import time
from helpers import transmon, calculate_geff, calc_average_transmon, gaussian_ramp_envelope
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


flux_params = {
    "As": (0.332, 0.0),
    "freqs": (0.275, 3*0.275), # GHz
    "phases": (0.0, 0.25), # rad/(2pi) 
}

transmon_params = {
    "fmax": 8.0, # GHz
    "d": 0.454, # SQUID asymmetry
    "alpha": 0.2, # - anharmonicity, GHz
} #GF/2 = 7.048551, GE= 7.15175

f_avg, alpha, lambda01, lambda12 = calc_average_transmon(transmon_params, flux_params)

drive_params = {
    "A": 0.015, # GHz
    "freq": f_avg-alpha/2, # GHz 
}

drive_pulse_params = {
    "t0": 0,
    "pulse_len": 5000,
    "ramp_std": 10,
    "ramp_chop": 2,
}

rr_params = {
    "freq": (f_avg-alpha) + 2*flux_params["freqs"][0], # GHz
    "trunc": 2,
    "g": 0.005, # coupling, GHz
    "kappa": 1/30, # GHz
}


# Preliminary calculations for drive Hamiltonian
s01 = qutip.basis(3, 0)*qutip.basis(3, 1).dag()
s12 = qutip.basis(3, 1)*qutip.basis(3, 2).dag()
delta_01 = drive_params["freq"] - f_avg
delta_12 = drive_params["freq"] - (f_avg - alpha) 
g_eff = calculate_geff(transmon_params, flux_params, Ns=[0])[0]
drive_env = gaussian_ramp_envelope(**drive_pulse_params)

def H_drive(t, *args):
    # pulse_params_new = pulse_params.copy() # Copy dictionary to set drive_len_value
    # pulse_params_new["pulse_len"] = args[0]["sweep_param"]
    drive_op = g_eff*(np.exp(1j*delta_01*t)*lambda01*s01 + np.exp(1j*delta_12*t)*np.sqrt(2)*lambda12*s12)
    H = 2*np.pi*drive_env(t)*drive_params["A"]*drive_op

    return H + H.dag()

# Preliminary calculations for coupling Hamiltonian
a = qutip.destroy(rr_params["trunc"])
delta_01_rr = rr_params["freq"] - f_avg
delta_12_rr = rr_params["freq"] - (f_avg - alpha) 
g_eff_2 = calculate_geff(transmon_params, flux_params, Ns=[2])[0]

def H_coupling(t, *args):
    # pulse_params_new = pulse_params.copy() # Copy dictionary to set drive_len_value
    # pulse_params_new["pulse_len"] = args[0]["sweep_param"]
    coupling_op = g_eff_2*(np.exp(1j*delta_01_rr*t)*lambda01*s01 + np.exp(1j*delta_12_rr*t)*np.sqrt(2)*lambda12*s12)
    coupling_op = qutip.tensor(coupling_op, a.dag())
    H = 2*np.pi*rr_params["g"]*coupling_op*np.exp(-1j*2*min(flux_params["freqs"])*t)

    return H + H.dag()

def H_total(t, *args):
    return qutip.tensor(H_drive(t, *args), qutip.qeye(rr_params["trunc"])) + H_coupling(t, *args)

initial_state = qutip.tensor(qutip.basis(3, 0), qutip.basis(rr_params["trunc"], 0))
t_list = np.arange(0, 5000, 1)
c_ops = [np.sqrt(rr_params["kappa"])*qutip.tensor(qutip.qeye(3), qutip.destroy(rr_params["trunc"]))]

start_time = time.time()  # Start timer
result = qutip.mesolve(H_total, initial_state, t_list, c_ops = c_ops, args = {})
print(f"Elapsed time: {time.time() - start_time:.6f} seconds")

pop0 = [np.real((qutip.ket2dm(qutip.basis(3, 0))*state.ptrace(0)).tr()) for state in result.states]
pop1 = [np.real((qutip.ket2dm(qutip.basis(3, 1))*state.ptrace(0)).tr()) for state in result.states]
pop2 = [np.real((qutip.ket2dm(qutip.basis(3, 2))*state.ptrace(0)).tr()) for state in result.states]


plt.plot(pop0, label = '0')
plt.plot(pop1, label = '1')
plt.plot(pop2, label = '2')

plt.grid()
plt.ylabel("Ground state population")
plt.xlabel("Time")
plt.legend()
plt.show()