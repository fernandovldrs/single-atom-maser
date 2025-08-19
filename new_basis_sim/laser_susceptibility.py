import qutip
import matplotlib.pyplot as plt
import numpy as np
import time
from helpers import transmon, calculate_geff, calc_average_transmon, gaussian_ramp_envelope
from multiprocessing import Pool
import time
import scipy.sparse as sp

###########################################################################
##                                                                       ##
## This script simulates the dynamics of the qubit as a function of      ##
## time under parametric modulation.                                     ##
## I'm doing this simulation in the charge basis, which considers        ##
## Non-adiabatic transitions and changes to driving parameters.          ##
##                                                                       ##
###########################################################################

transmon_trunc = 3

flux_params = {
    "As": (0.33242828172118893, 0.0),
    "freqs": (0.275, 3*0.275), # GHz
    "phases": (0.0, 0.25), # rad/(2pi) 
}

transmon_params = {
    "fmax": 8.0, # GHz
    "d": 0.454, # SQUID asymmetry
    "alpha": 0.2, # - anharmonicity, GHz
} #GF/2 = 7.048551, GE= 7.15175

f_avg, alpha, lambda01, lambda12, xi_avg = calc_average_transmon(transmon_params, flux_params)

drive_params = {
    "A": 0.0402, # GHz
    "freq": f_avg-alpha/2, # GHz 
}

rr_params = {
    "freq": (f_avg-alpha) + 2*flux_params["freqs"][0], # GHz
    "trunc": 2,
    "g": 0.030, # coupling, GHz
    "kappa": 1/25, # GHz
}

cav_params = {
    "freq": f_avg - 2*flux_params["freqs"][0], # GHz
    "trunc": 40,
    "g": 0.011, # coupling, GHz
    "kappa": 1/50000, # GHz
}

perturbation_params = {
    "A": 0.002, # GHz
    "freq": cav_params["freq"], # GHz 
}


g2, g0, gm2 = calculate_geff(transmon_params, flux_params, Ns=[2, 0, -2])

# Preliminary calculations for drive Hamiltonian
p11 = qutip.basis(transmon_trunc, 1)*qutip.basis(transmon_trunc, 1).dag()
p22 = qutip.basis(transmon_trunc, 2)*qutip.basis(transmon_trunc, 2).dag()
s01 = qutip.basis(transmon_trunc, 0)*qutip.basis(transmon_trunc, 1).dag()
s12 = qutip.basis(transmon_trunc, 1)*qutip.basis(transmon_trunc, 2).dag()
def H_drive():
    drive_op = g0*(lambda01*s01 + np.sqrt(2)*lambda12*s12)
    H = 2*np.pi*drive_params["A"]*drive_op

    return H + H.dag()

a = qutip.destroy(rr_params["trunc"])
def H_rr_coupling():
    coupling_op = g2*(lambda01*s01 + np.sqrt(2)*lambda12*s12)
    coupling_op = qutip.tensor(coupling_op, a.dag())
    H = 2*np.pi*rr_params["g"]*coupling_op
    return H + H.dag()

b = qutip.destroy(cav_params["trunc"])
def H_cav_coupling():
    coupling_op = gm2*(lambda01*s01 + np.sqrt(2)*lambda12*s12)
    coupling_op = qutip.tensor(b.dag(), coupling_op)
    H = 2*np.pi*cav_params["g"]*coupling_op
    return H + H.dag()

def H_perturbation(t, *args):
    f = cav_params["freq"] - drive_params["freq"] + 2*flux_params["freqs"][0]
    drive_op = b*np.exp(-1j*2*np.pi*f*t)
    H = 2*np.pi*drive_params["A"]*drive_op

    return H + H.dag()

def H_total(t, *args):
    
    Ht = 2*np.pi*(f_avg - drive_params["freq"])*(p11 + 2*p22)
    Hr = 2*np.pi*(rr_params["freq"] - drive_params["freq"] - 2*flux_params["freqs"][0])*a.dag()*a
    Hc = 2*np.pi*(cav_params["freq"] - drive_params["freq"] + 2*flux_params["freqs"][0])*b.dag()*b
    Hp = H_perturbation(t, *args)

    Ht = qutip.tensor(qutip.qeye(cav_params["trunc"]), Ht, qutip.qeye(rr_params["trunc"]))
    Hr = qutip.tensor(qutip.qeye(cav_params["trunc"]), qutip.qeye(transmon_trunc), Hr)
    Hc = qutip.tensor(Hc, qutip.qeye(transmon_trunc), qutip.qeye(rr_params["trunc"]))
    Hp = qutip.tensor(Hp, qutip.qeye(transmon_trunc), qutip.qeye(rr_params["trunc"]))
    
    # Interaction Hamiltonians
    H = qutip.tensor(qutip.qeye(cav_params["trunc"]), H_drive(), qutip.qeye(rr_params["trunc"]))
    H += qutip.tensor(qutip.qeye(cav_params["trunc"]), H_rr_coupling())
    H += qutip.tensor(H_cav_coupling(), qutip.qeye(rr_params["trunc"]))
    return qutip.Qobj(sp.csr_matrix( (Ht + Hr + Hc + Hp + H).full(), dtype=complex), dims=[[40, 3, 2], [40, 3, 2]])
    # return 

c_ops = [np.sqrt(rr_params["kappa"])*qutip.tensor(qutip.qeye(cav_params["trunc"]), 
                                                  qutip.qeye(transmon_trunc), 
                                                  a),
        np.sqrt(cav_params["kappa"])*qutip.tensor(b, 
                                                  qutip.qeye(transmon_trunc), 
                                                  qutip.qeye(rr_params["trunc"])),]



# initial_state = qutip.tensor(qutip.basis(cav_params["trunc"], 0), 
#                              qutip.basis(transmon_trunc, 0),
#                              qutip.basis(rr_params["trunc"], 0))

filename = f'data/laser_threshold_0/bistability/state_402MHz.npz'
# filename = f'perturbation.npz'
loaded_data = np.load(filename)
dims = loaded_data['dims'].tolist()
initial_state = qutip.Qobj(loaded_data['data'], dims=dims)

t_list = np.arange(0, 15000, 5)
start_time = time.time()  

# opts = qutip.Options(store_states=False, sparse=True)
result = qutip.mesolve(H_total, initial_state, t_list, c_ops = c_ops)
print(f"Elapsed time: {time.time() - start_time:.6f} seconds")
final_state = result.states[-1]

# Convert Qobj to NumPy array
filename = "perturbation2"
array = final_state.full()
print("Saving temporary "+ filename)
np.savez(filename, data=array, dims=final_state.dims)
