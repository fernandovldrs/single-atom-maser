import dynamiqs as dq
import numpy as np
import matplotlib.pyplot as plt
import time
import jax.numpy as jnp

start_time = time.time()

# Simulation parameters
res_trunc = 20
aux_trunc = 2
transmon_trunc = 3

t_sim = 20000
timestep = 4
t_list = jnp.arange(0, t_sim, timestep)

# Create destruction operators
a = dq.operators.destroy(res_trunc)
b = dq.operators.destroy(aux_trunc)
c = dq.operators.destroy(transmon_trunc)

# Tensor product to match system dimensions
a = dq.tensor(a, dq.eye(aux_trunc), dq.eye(transmon_trunc))
b = dq.tensor(dq.eye(res_trunc), b, dq.eye(transmon_trunc))
c = dq.tensor(dq.eye(res_trunc), dq.eye(aux_trunc), c)

# System parameters
wge = 6
alpha = -200e-3
waux = 5.8
wgf2 = (wge + waux)/2 
g_res = 6.5e-3  # 10MHz
g_aux = 23.5e-3  # 30MHz
omega_gf2 = 25e-3  # 20MHz
kappa_res = 0.31e-3  # T1 = 100us
kappa_aux = 138e-3  # T1 = 300ns

# Define Hamiltonian
H = g_res * (dq.dag(a) @ c + a @ dq.dag(c))
H += g_aux * (dq.dag(b) @ c + b @ dq.dag(c))
H += omega_gf2/2 * (c + dq.dag(c))
H += (wge - wgf2) * dq.dag(a)@a
H += (waux - wgf2) * dq.dag(b)@b
H += (wge - wgf2) * dq.dag(c)@c + alpha/2 * dq.dag(c)@c@(dq.dag(c)@c - dq.eye(*c.dims) )
H*= 2*jnp.pi
print(H)

# Dissipation
c_ops = [jnp.sqrt(kappa_res) * a, jnp.sqrt(kappa_aux) * b]

# Initial state
initial_state = dq.tensor(dq.fock(res_trunc, 0),
                               dq.fock(aux_trunc, 0),
                               dq.fock(transmon_trunc, 0))

# Simulation
result = dq.mesolve(H, c_ops, initial_state, t_list)#, options={"nsteps": 4000})
final_state = result.states[-1]


mean_n = dq.tracemm(dq.dag(a)@a, final_state)
mean_n_squared = dq.tracemm(dq.dag(a)@a@dq.dag(a)@a, final_state)
variance_n = mean_n_squared - mean_n**2
fano_number = variance_n / mean_n
print(mean_n, fano_number)


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

# extremity = max([jnp.abs(jnp.max(W_t)), np.abs(jnp.min(W_t))])
# ax[1].pcolormesh(x, p, W_t.T, cmap="bwr", vmin=-extremity, vmax=extremity)
# ax[1].set_title("Wigner function")
# ax[1].set_xlabel("Re[β]")
# ax[1].set_ylabel("Im[β]")

# Save figure
filename = "final_state_{}.png".format(int(t_sim))
plt.savefig(filename, dpi=150, bbox_inches='tight')

print("Simulation time:", time.time() - start_time)
