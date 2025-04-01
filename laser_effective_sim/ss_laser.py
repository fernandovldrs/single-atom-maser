import qutip
from qutip.solver import Options
import matplotlib.pyplot as plt
import numpy as np
from qutip import wigner
import time
from helpers import *
import scipy.sparse as sp

###########################################################################
##                                                                       ##
## This script finds the steady-state solution of the laser dynamics.    ##
## It matches with previous theory in the literature.                    ##
## The drive interaction and capacitive couplings are derived            ##
## from the charge basis of the transmon.                                ##
##                                                                       ##
###########################################################################

# Simulation parameters
res_trunc = 120
transmon_trunc = 4
aux_trunc = 2
dims = [res_trunc, transmon_trunc, aux_trunc]
d_total = res_trunc*transmon_trunc*aux_trunc

# System parameters
fge = 6600
alpha = -200
faux = alpha + fge
# wgf2 = (fge + faux)/2
g_res = 11*0.5653  # 10MHz
g_aux = 30*0.5535  # 30MHz
omega_gf2 = 25  # 20MHz
kappa_res = 0.01/2  # T1 = 100us
kappa_aux = 3.33  # T1 = 300ns
gamma_tr = 0.1  # T1 = 10us

# Define Hamiltonian and losses
trs =  transmon(f_ge = fge, alpha = alpha, g_ef = g_aux, g_ge = g_res, 
                       gamma_res = kappa_res, kappa = kappa_aux, n_ph = res_trunc, f_q = omega_gf2, 
                       n_trunc = transmon_trunc, gamma_tr = gamma_tr)
H = trs.build_H()
c_ops = trs.build_C()
H = qutip.Qobj(sp.csr_matrix(H.full(), dtype=complex))

# Simulation
# final_state = qutip.steadystate(H, c_ops, method = 'power', use_rcm = True)
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

# Plot results
fig, ax = plt.subplots(1, 2, figsize = (7*0.9,5*0.9), constrained_layout=True)

photon_distribution = []
for level in range(res_trunc):
    proj = qutip.tensor(qutip.basis(res_trunc, level)*qutip.basis(res_trunc, level).dag(),
                        qutip.qeye(transmon_trunc),
                        qutip.qeye(aux_trunc)) 
    level_pop = (proj*final_state).tr()
    photon_distribution.append(level_pop)

p_list = []
for level in range(transmon_trunc): 
    proj = qutip.tensor(qutip.qeye(res_trunc),
                        qutip.basis(transmon_trunc, level)*qutip.basis(transmon_trunc, level).dag(),
                        qutip.qeye(aux_trunc)) 
    level_pop = (proj*final_state).tr()
    p_list.append(level_pop)

print("Qubit population: ", p_list)

x = np.linspace(-10, 10, 251)
p = np.linspace(-10, 10, 251)
W_t = wigner(final_state.ptrace(0), x, p)
extremety = max([np.abs(np.max(W_t)), np.abs(np.min(W_t))])
font_size = 16
border_linewidth = 2
ax[0].bar(range(res_trunc), photon_distribution, color='blue', alpha=0.7)
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
