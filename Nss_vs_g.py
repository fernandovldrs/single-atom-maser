import dynamiqs as dq
import numpy as np
import matplotlib.pyplot as plt
import time
import jax.numpy as jnp

# Simulation parameters
res_trunc = 70
aux_trunc = 2
transmon_trunc = 3

t_sim = 160000
timestep = 50
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

# Initial state
initial_state = dq.tensor(dq.fock(res_trunc, 0),
                            dq.fock(aux_trunc, 0),
                            dq.fock(transmon_trunc, 0))

fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
g_scaling_list = jnp.linspace(0, 4, 16)
Nss_list = []
F_list = []
for g_scaling in g_scaling_list:
    # System parameters
    g_res = 11e-3*0.04103*g_scaling  # 10MHz
    g_aux = 30e-3*0.0533*g_scaling  # 30MHz
    omega_gf2 = g_res + g_aux
    kappa_res = 0.033e-3 # T1 = 100us/3
    kappa_aux = 3.33e-3  # T1 = 300ns

    # Define Hamiltonian
    H = g_res * (dq.dag(a) @ sge + a @ dq.dag(sge))
    H += g_aux * (dq.dag(b) @ sef + b @ dq.dag(sef))
    H += omega_gf2 * (sge @ sef + dq.dag(sef) @ dq.dag(sge))
    H *= 2*jnp.pi

    # Dissipation
    c_ops = [jnp.sqrt(kappa_res) * a, jnp.sqrt(kappa_aux) * b]

    # Simulation
    result = dq.mesolve(H, c_ops, initial_state, t_list)

    final_state = result.states[-1]
    mean_n = dq.tracemm(dq.dag(a)@a, final_state)
    mean_n_squared = dq.tracemm(dq.dag(a)@a@dq.dag(a)@a, final_state)
    variance_n = mean_n_squared - mean_n**2
    fano_number = variance_n / mean_n
    print(mean_n, fano_number)

    Nss_list.append(mean_n)
    F_list.append(fano_number)

ax[0].plot(g_scaling_list*1e3, Nss_list, color='blue', alpha=0.7)
ax[0].set_xlabel("2-photon pump (MHz)")
ax[0].set_ylabel("Nss")

ax[1].plot(g_scaling_list*1e3, F_list, color='blue', alpha=0.7)
ax[1].set_xlabel("2-photon pump (MHz)")
ax[1].set_ylabel("Nss")

plt.show()