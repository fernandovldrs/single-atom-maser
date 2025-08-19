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
    "As": (0.332, 0.0),
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
    "A": 0.090, # GHz
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
    "trunc": 200,
    "g": 0.011, # coupling, GHz
    "kappa": 1/50000, # GHz
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

def H_total():
    
    Ht = 2*np.pi*(f_avg - drive_params["freq"])*(p11 + 2*p22)
    Hr = 2*np.pi*(rr_params["freq"] - drive_params["freq"] - 2*flux_params["freqs"][0])*a.dag()*a
    Hc = 2*np.pi*(cav_params["freq"] - drive_params["freq"] + 2*flux_params["freqs"][0])*b.dag()*b

    Ht = qutip.tensor(qutip.qeye(cav_params["trunc"]), Ht, qutip.qeye(rr_params["trunc"]))
    Hr = qutip.tensor(qutip.qeye(cav_params["trunc"]), qutip.qeye(transmon_trunc), Hr)
    Hc = qutip.tensor(Hc, qutip.qeye(transmon_trunc), qutip.qeye(rr_params["trunc"]))
    
    # Interaction Hamiltonians
    H = qutip.tensor(qutip.qeye(cav_params["trunc"]), H_drive(), qutip.qeye(rr_params["trunc"]))
    H += qutip.tensor(qutip.qeye(cav_params["trunc"]), H_rr_coupling())
    H += qutip.tensor(H_cav_coupling(), qutip.qeye(rr_params["trunc"]))
    return Ht + Hr + Hc + H

c_ops = [np.sqrt(rr_params["kappa"])*qutip.tensor(qutip.qeye(cav_params["trunc"]), 
                                                  qutip.qeye(transmon_trunc), 
                                                  a),
        np.sqrt(cav_params["kappa"])*qutip.tensor(b, 
                                                  qutip.qeye(transmon_trunc), 
                                                  qutip.qeye(rr_params["trunc"])),]


H = qutip.Qobj(sp.csr_matrix(H_total().full(), dtype=complex))

start_time = time.time()  # Start timer
final_state = qutip.steadystate(H, c_ops, method = 'iterative')
print(f"Elapsed time: {time.time() - start_time:.6f} seconds")

array = final_state.full()
reshaped_array = array.reshape((cav_params["trunc"], transmon_trunc, rr_params["trunc"], 
                                cav_params["trunc"], transmon_trunc, rr_params["trunc"]))
d_total = cav_params["trunc"]*transmon_trunc*rr_params["trunc"]
reshaped_array = reshaped_array.reshape((d_total, d_total))

# Convert back to Qobj
final_state = qutip.Qobj(reshaped_array, dims=[[cav_params["trunc"], transmon_trunc, rr_params["trunc"]], 
                                               [cav_params["trunc"], transmon_trunc, rr_params["trunc"]]])

# Plot results
fig, ax = plt.subplots(1, 2, figsize = (7*0.9,5*0.9), constrained_layout=True)

photon_distribution = []
for level in range(cav_params["trunc"]):
    proj = qutip.tensor(qutip.basis(cav_params["trunc"], level)*qutip.basis(cav_params["trunc"], level).dag(),
                        qutip.qeye(transmon_trunc),
                        qutip.qeye(rr_params["trunc"])) 
    level_pop = (proj*final_state).tr()
    photon_distribution.append(level_pop)

p_list = []
for level in range(transmon_trunc): 
    proj = qutip.tensor(qutip.qeye(cav_params["trunc"]),
                        qutip.basis(transmon_trunc, level)*qutip.basis(transmon_trunc, level).dag(),
                        qutip.qeye(rr_params["trunc"])) 
    level_pop = (proj*final_state).tr()
    p_list.append(level_pop)

print("Qubit population: ", p_list)

filename = f'final_state'
# Convert Qobj to NumPy array
array = final_state.full()
print("Saving "+ filename)
np.savez(filename, data=array, dims=final_state.dims)

x = np.linspace(-10, 10, 251)
p = np.linspace(-10, 10, 251)
W_t = qutip.wigner(final_state.ptrace(0), x, p)
extremety = max([np.abs(np.max(W_t)), np.abs(np.min(W_t))])
font_size = 16
border_linewidth = 2
ax[0].bar(range(cav_params["trunc"]), photon_distribution, color='blue', alpha=0.7)
ax[0].set_title("Photon number distribution")
ax[0].set_xlabel("Fock state")
ax[0].set_ylabel("Population")
ax[0].set_ylim([0.0, 1.0])
ax[1].pcolormesh(x, p, W_t.T, cmap = "bwr", vmin = -extremety, vmax = extremety)
ax[1].set_title(r"Wigner function", fontsize=font_size)
ax[1].set_xlabel(r'Re[$\beta$]', fontsize=font_size)
ax[1].set_ylabel(r'Im[$\beta$]', fontsize=font_size)
ax[1].tick_params(axis='both', width=border_linewidth, labelsize=font_size, direction='in', length=8)
plt.show()
