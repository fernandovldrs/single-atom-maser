import dynamiqs as dq
import numpy as np
import matplotlib.pyplot as plt
import time
import jax.numpy as jnp

# Simulation parameters
res_trunc = 200
aux_trunc = 2
transmon_trunc = 3

t_sim = 200000
timestep = 25
t_list = jnp.arange(0, t_sim, timestep)

# Create destruction operators
a = dq.operators.destroy(res_trunc)
b = dq.operators.destroy(aux_trunc)

# Define ladder operators for the transmon
sge = dq.asqarray([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
sef = dq.asqarray([[0, 0, 0], [0, 0, 1], [0, 0, 0]])

# Tensor product to match system dimensions
a = dq.tensor(a, dq.eye(aux_trunc), dq.eye(transmon_trunc))
b = dq.tensor(dq.eye(res_trunc), b, dq.eye(transmon_trunc))
sge = dq.tensor(dq.eye(res_trunc), dq.eye(aux_trunc), sge)
sef = dq.tensor(dq.eye(res_trunc), dq.eye(aux_trunc), sef)

# System parameters
g_scaling = 4
g_res = 11e-3*0.04103*g_scaling  # 10MHz
g_aux = 30e-3*0.0533*g_scaling  # 30MHz
omega_gf2 = g_aux + g_res*2 
print(omega_gf2*1e3)
kappa_res = 0.01e-3 # T1 = 100us
kappa_aux = 3.33e-3  # T1 = 300ns

# Define Hamiltonian
H = g_res * (dq.dag(a) @ sge + a @ dq.dag(sge))
H += g_aux * (dq.dag(b) @ sef + b @ dq.dag(sef))
H += omega_gf2 * (sge @ sef + dq.dag(sef) @ dq.dag(sge))
H *= 2*jnp.pi

# Dissipation
c_ops = [jnp.sqrt(kappa_res) * a, jnp.sqrt(kappa_aux) * b]

# Initial state
filename = "saved_state.npz"
np_initial_state = np.load(filename)
print(np_initial_state.files)
initial_state = dq.asqarray(np_initial_state["state"], dims = (res_trunc, aux_trunc, transmon_trunc))
# initial_state = dq.tensor(dq.fock(res_trunc, 0),
#                               dq.fock(aux_trunc, 0),
#                               dq.fock(transmon_trunc, 0))

# Simulation
result = dq.mesolve(H, c_ops, initial_state, t_list)

final_state = result.states[-1]

## Save final state
np_final_state = np.array(final_state)
filename = "saved_state.npz"
np.savez(filename, state = np_final_state)

alpha = dq.tracemm(a, final_state)
mean_n = dq.tracemm(dq.dag(a)@a, final_state)
mean_n_squared = dq.tracemm(dq.dag(a)@a@dq.dag(a)@a, final_state)
variance_n = mean_n_squared - mean_n**2
fano_number = variance_n / mean_n
print(mean_n, fano_number, alpha)

# Plot results
fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)

photon_distribution = []
for level in range(res_trunc):
    proj = dq.tensor(dq.fock(res_trunc, level) @ dq.dag(dq.fock(res_trunc, level)),
                             dq.eye(aux_trunc),
                             dq.eye(transmon_trunc))
    level_pop = (proj @ final_state).trace()
    photon_distribution.append(level_pop)

x = jnp.linspace(-10, 10, 301)
p = jnp.linspace(-10, 10, 301)
dq.plot.wigner(dq.ptrace(final_state, 0), ax=ax[1], xmax = 7, ymax = 7, npixels = 301)

ax[0].bar(range(res_trunc), photon_distribution, color='blue', alpha=0.7)
ax[0].set_title("Photon number distribution")
ax[0].set_xlabel("Fock state")
ax[0].set_ylabel("Population")
ax[0].set_ylim([0.0, 1.0])

fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
skip = 20
t_plot_list = np.arange(0, t_sim, timestep*skip)
avg_n_list = []
f_list = []
for state in result.states[::skip]:
    mean_n = dq.tracemm(dq.dag(a)@a, state)
    avg_n_list.append(mean_n)
    mean_n_squared = dq.tracemm(dq.dag(a)@a@dq.dag(a)@a, state)
    variance_n = mean_n_squared - mean_n**2
    fano_number = variance_n / mean_n
    f_list.append(fano_number)

plt.plot(t_plot_list, avg_n_list)
plt.plot(t_plot_list, f_list)
plt.show()
