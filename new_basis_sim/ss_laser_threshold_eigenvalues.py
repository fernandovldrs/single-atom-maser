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
    "A": 0.080, # GHz
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
    "trunc": [40], # change the truncation to i -> i+1 when necessary
    "g": 0.011, # coupling, GHz
    "kappa": 1/50000, # GHz
}

g2, g0, gm2 = calculate_geff(transmon_params, flux_params, Ns=[2, 0, -2])

p11 = qutip.basis(transmon_trunc, 1)*qutip.basis(transmon_trunc, 1).dag()
p22 = qutip.basis(transmon_trunc, 2)*qutip.basis(transmon_trunc, 2).dag()
s01 = qutip.basis(transmon_trunc, 0)*qutip.basis(transmon_trunc, 1).dag()
s12 = qutip.basis(transmon_trunc, 1)*qutip.basis(transmon_trunc, 2).dag()

a = qutip.destroy(rr_params["trunc"])
def H_rr_coupling():
    coupling_op = g2*(lambda01*s01 + np.sqrt(2)*lambda12*s12)
    coupling_op = qutip.tensor(coupling_op, a.dag())
    H = 2*np.pi*rr_params["g"]*coupling_op
    return H + H.dag()

## Initial cavity truncation
cav_trunc_indx = 0
cav_trunc = cav_params["trunc"][cav_trunc_indx]

# Sweep over drive strength
drive_A_list = np.arange(0.038, 0.044, 0.0005)
for drive_A in drive_A_list:
    drive_params_new = drive_params.copy()
    drive_params_new["A"] = drive_A

    pop = 1
    while pop > 0.01: 

        cav_trunc = cav_params["trunc"][cav_trunc_indx]
        dims = [[cav_trunc, transmon_trunc, rr_params["trunc"]], 
                [cav_trunc, transmon_trunc, rr_params["trunc"]]]
        d_total = cav_trunc*transmon_trunc*rr_params["trunc"]

        def H_drive():
            drive_op = g0*(lambda01*s01 + np.sqrt(2)*lambda12*s12)
            H = 2*np.pi*drive_params_new["A"]*drive_op

            return H + H.dag()

        b = qutip.destroy(cav_trunc)
        def H_cav_coupling():
            coupling_op = gm2*(lambda01*s01 + np.sqrt(2)*lambda12*s12)
            coupling_op = qutip.tensor(b.dag(), coupling_op)
            H = 2*np.pi*cav_params["g"]*coupling_op
            return H + H.dag()

        def H_total():
            
            Ht = 2*np.pi*(f_avg - drive_params_new["freq"])*(p11 + 2*p22)
            Hr = 2*np.pi*(rr_params["freq"] - drive_params_new["freq"] - 2*flux_params["freqs"][0])*a.dag()*a
            Hc = 2*np.pi*(cav_params["freq"] - drive_params_new["freq"] + 2*flux_params["freqs"][0])*b.dag()*b

            Ht = qutip.tensor(qutip.qeye(cav_trunc), Ht, qutip.qeye(rr_params["trunc"]))
            Hr = qutip.tensor(qutip.qeye(cav_trunc), qutip.qeye(transmon_trunc), Hr)
            Hc = qutip.tensor(Hc, qutip.qeye(transmon_trunc), qutip.qeye(rr_params["trunc"]))
            
            # Interaction Hamiltonians
            H = qutip.tensor(qutip.qeye(cav_trunc), H_drive(), qutip.qeye(rr_params["trunc"]))
            H += qutip.tensor(qutip.qeye(cav_trunc), H_rr_coupling())
            H += qutip.tensor(H_cav_coupling(), qutip.qeye(rr_params["trunc"]))
            return Ht + Hr + Hc + H

        c_ops = [np.sqrt(rr_params["kappa"])*qutip.tensor(qutip.qeye(cav_trunc), 
                                                        qutip.qeye(transmon_trunc), 
                                                        a),
                np.sqrt(cav_params["kappa"])*qutip.tensor(b, 
                                                        qutip.qeye(transmon_trunc), 
                                                        qutip.qeye(rr_params["trunc"])),]


        H = qutip.Qobj(sp.csr_matrix(H_total().full(), dtype=complex))

        start_time = time.time()  # Start timer
        # final_state = qutip.steadystate(H, c_ops, method = 'iterative')
        L = qutip.liouvillian(H, c_ops)
        L = qutip.Qobj(sp.csr_matrix(qutip.liouvillian(H, c_ops).full(), dtype=complex))
        evals = L.eigenenergies(sort='low')
        print(evals[:8])
        print(f"Elapsed time: {time.time() - start_time:.6f} seconds")
  