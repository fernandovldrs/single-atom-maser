import qutip
from qutip.solver import Options
import matplotlib.pyplot as plt
import numpy as np
from qutip import wigner
import time

# Simulation parameters
res_trunc = 20
aux_trunc = 2
transmon_trunc = 3
dims = [res_trunc, aux_trunc, transmon_trunc]
d_total = res_trunc*aux_trunc*transmon_trunc

# Destruction operators
a = qutip.destroy(res_trunc)
b = qutip.destroy(aux_trunc)
c = qutip.destroy(transmon_trunc)

a = qutip.tensor(a, qutip.qeye(aux_trunc), qutip.qeye(transmon_trunc))
b = qutip.tensor(qutip.qeye(res_trunc), b, qutip.qeye(transmon_trunc))
c = qutip.tensor(qutip.qeye(res_trunc),  qutip.qeye(aux_trunc), c)

# System parameters
wge = 0
alpha = -200
waux = alpha
wgf2 = (wge + waux)/2 
g_res = 11 # 10MHz
g_aux = 8  # 30MHz
omega_gf2 = 25  # 20MHz
kappa_res = 0.2  # T1 = 100us
kappa_aux = 138  # T1 = 300ns

# Define Hamiltonian 1
H_1 = g_res*(a.dag()*c + a*c.dag()) 
H_1 += g_aux * (b.dag() * c + b * c.dag())
H_1 += omega_gf2/2 * (c + c.dag())
H_1 += (wge - wgf2) * a.dag()*a
H_1 += (waux - wgf2) * b.dag()*b
H_1 += (wge - wgf2) * c.dag()*c + alpha/2 * c.dag()*c*(c.dag()*c - 1 )
H_1 *= 2*np.pi

# Define ladder operators for the transmon
sge = qutip.Qobj(np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
sef = qutip.Qobj(np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]))
sge = qutip.tensor(qutip.qeye(res_trunc), qutip.qeye(aux_trunc), sge)
sef = qutip.tensor(qutip.qeye(res_trunc), qutip.qeye(aux_trunc), sef)

# Define Hamiltonian 2
H_2 = g_res * (a.dag() * sge + a * sge.dag())
H_2 += g_aux * np.sqrt(2) * (b.dag() * sef + b * sef.dag())
H_2 += -omega_gf2**2/np.sqrt(2)/alpha * (sge * sef + sef.dag() * sge.dag())
H_2 *= 2*np.pi

# Losses
c_ops = [np.sqrt(kappa_res) * a, np.sqrt(kappa_aux) * b]

# Simulation
initial_state = qutip.tensor(qutip.basis(res_trunc, 0), 
                                qutip.basis(aux_trunc, 0), 
                                qutip.basis(transmon_trunc, 0))

L_1 = qutip.liouvillian(H_1, c_ops)
L_2 = qutip.liouvillian(H_2, c_ops)

t1 = time.time()
final_state_1 = qutip.steadystate(L_1, method = 'power', use_rcm = True)
t2 = time.time()
print(t2-t1)
final_state_2 = qutip.steadystate(L_2, method = 'power', use_rcm = True)
t3 = time.time()
print(t3-t2)
print("State overlap: ", (final_state_1.unit()).overlap(final_state_2.unit()))

# Plot results
fig, ax = plt.subplots(2, 2, figsize = (10*0.9,5*0.9), constrained_layout=True)

final_state = final_state_1.unit()
photon_distribution = []
for level in range(res_trunc):
    proj = qutip.tensor(qutip.basis(res_trunc, level)*qutip.basis(res_trunc, level).dag(), 
                        qutip.qeye(aux_trunc),
                        qutip.qeye(transmon_trunc)) 
    level_pop = (proj*final_state).tr()
    photon_distribution.append(level_pop)

x = np.linspace(-10, 10, 251)
p = np.linspace(-10, 10, 251)
W_t = wigner(final_state.ptrace(0), x, p)
extremety = max([np.abs(np.max(W_t)), np.abs(np.min(W_t))])
font_size = 16
border_linewidth = 2
ax[0][1].pcolormesh(x, p, W_t.T, cmap = "bwr", vmin = -extremety, vmax = extremety)
ax[0][1].set_title(r"Wigner function", fontsize=font_size)
ax[0][1].set_xlabel(r'Re[$\beta$]', fontsize=font_size)
ax[0][1].set_ylabel(r'Im[$\beta$]', fontsize=font_size)
ax[0][1].tick_params(axis='both', width=border_linewidth, labelsize=font_size, direction='in', length=8)
ax[0][0].bar(range(res_trunc), photon_distribution, color='blue', alpha=0.7)
ax[0][0].set_title("Photon number distribution")
ax[0][0].set_xlabel("Fock state")
ax[0][0].set_ylabel("Population")
ax[0][0].set_ylim([0.0, 1.0])

final_state = final_state_2.unit()
photon_distribution = []
for level in range(res_trunc):
    proj = qutip.tensor(qutip.basis(res_trunc, level)*qutip.basis(res_trunc, level).dag(), 
                        qutip.qeye(aux_trunc),
                        qutip.qeye(transmon_trunc)) 
    level_pop = (proj*final_state).tr()
    photon_distribution.append(level_pop)

x = np.linspace(-10, 10, 251)
p = np.linspace(-10, 10, 251)
W_t = wigner(final_state.ptrace(0), x, p)
extremety = max([np.abs(np.max(W_t)), np.abs(np.min(W_t))])
font_size = 16
border_linewidth = 2
ax[1][1].pcolormesh(x, p, W_t.T, cmap = "bwr", vmin = -extremety, vmax = extremety)
ax[1][1].set_title(r"Wigner function", fontsize=font_size)
ax[1][1].set_xlabel(r'Re[$\beta$]', fontsize=font_size)
ax[1][1].set_ylabel(r'Im[$\beta$]', fontsize=font_size)
ax[1][1].tick_params(axis='both', width=border_linewidth, labelsize=font_size, direction='in', length=8)
ax[1][0].bar(range(res_trunc), photon_distribution, color='blue', alpha=0.7)
ax[1][0].set_title("Photon number distribution")
ax[1][0].set_xlabel("Fock state")
ax[1][0].set_ylabel("Population")
ax[1][0].set_ylim([0.0, 1.0])

plt.show()
