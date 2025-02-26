import qutip
from qutip.solver import Options
import matplotlib.pyplot as plt
import numpy as np
from qutip import wigner

# Simulation parameters
res_trunc = 15
aux_trunc = 2
transmon_trunc = 3

t_sim = 20000
timestep = 2
t_list = np.arange(0, t_sim, timestep)

# Destruction operators
a = qutip.destroy(res_trunc)
b = qutip.destroy(aux_trunc)
c = qutip.destroy(transmon_trunc)

a = qutip.tensor(a, qutip.qeye(aux_trunc), qutip.qeye(transmon_trunc))
b = qutip.tensor(qutip.qeye(res_trunc), b, qutip.qeye(transmon_trunc))
c = qutip.tensor(qutip.qeye(res_trunc),  qutip.qeye(aux_trunc), c)

# System parameters
wge = 0
alpha = -200e-3
waux = alpha
wgf2 = (wge + waux)/2 
g_res = 8e-3  # 10MHz
g_aux = 15e-3  # 30MHz
omega_gf2 = 25e-3  # 20MHz
kappa_res = 0.2e-3  # T1 = 100us
kappa_aux = 138e-3  # T1 = 300ns

# Define Hamiltonian
H = g_res*(a.dag()*c + a*c.dag()) 
H += g_aux * (b.dag() * c + b * c.dag())
H += omega_gf2/2 * (c + c.dag())
H += (wge - wgf2) * a.dag()*a
H += (waux - wgf2) * b.dag()*b
H += (wge - wgf2) * c.dag()*c + alpha/2 * c.dag()*c*(c.dag()*c - 1 )
H *= 2*np.pi


# Losses
c_ops = [np.sqrt(kappa_res) * a, np.sqrt(kappa_aux) * b]

# Simulation
initial_state = qutip.tensor(qutip.basis(res_trunc, 0), 
                                qutip.basis(aux_trunc, 0), 
                                qutip.basis(transmon_trunc, 0))

result = qutip.mesolve(H, initial_state, t_list, c_ops = c_ops, options=Options(nsteps=4000))
final_state = result.states[-1]


# Plot results
fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)

photon_distribution = []
fig, ax = plt.subplots(1, 2, figsize = (10*0.9,5*0.9), constrained_layout=True)
final_state = final_state.unit()
photon_distribution = []
for level in range(res_trunc):
    proj = qutip.tensor(qutip.basis(res_trunc, level)*qutip.basis(res_trunc, level).dag(), 
                        qutip.qeye(aux_trunc),
                        qutip.qeye(transmon_trunc)) 
    level_pop = (proj*final_state).tr()
    photon_distribution.append(level_pop)

x = np.linspace(-10, 10, 301)
p = np.linspace(-10, 10, 301)
W_t = wigner(final_state.ptrace(0), x, p)
extremety = max([np.abs(np.max(W_t)), np.abs(np.min(W_t))])
font_size = 16
border_linewidth = 2
ax[1].pcolormesh(x, p, W_t.T, cmap = "bwr", vmin = -extremety, vmax = extremety)
ax[1].set_title(r"Wigner function", fontsize=font_size)
ax[1].set_xlabel(r'Re[$\beta$]', fontsize=font_size)
ax[1].set_ylabel(r'Im[$\beta$]', fontsize=font_size)
ax[1].tick_params(axis='both', width=border_linewidth, labelsize=font_size, direction='in', length=8)
ax[0].bar(range(res_trunc), photon_distribution, color='blue', alpha=0.7)
ax[0].set_title("Photon number distribution")
ax[0].set_xlabel("Fock state")
ax[0].set_ylabel("Population")
ax[0].set_ylim([0.0, 1.0])

fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
skip = 1
t_plot_list = np.arange(0, t_sim, timestep*skip)
avg_n_list = []
f_list = []
for state in result.states[5::skip]:
    mean_n = (a.dag()*a*state).tr()
    avg_n_list.append(mean_n)
    mean_n_squared = qutip.expect((a.dag()*a)**2, state)
    variance_n = mean_n_squared - mean_n**2
    fano_number = variance_n / mean_n
    f_list.append(fano_number)

print((a.dag()*a*final_state).tr())
plt.plot(t_plot_list[5::skip], avg_n_list)
plt.plot(t_plot_list[5::skip], f_list)
plt.show()
