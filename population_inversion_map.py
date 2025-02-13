import dynamiqs as dq
import numpy as np
import matplotlib.pyplot as plt
import time
import jax.numpy as jnp

# Simulation parameters
aux_trunc = 2
transmon_trunc = 3

t_sim = 30000
timestep = 50
t_list = jnp.arange(0, t_sim, timestep)

# Create destruction operators
b = dq.operators.destroy(aux_trunc)

# Define ladder operators for the transmon
sge = dq.asqarray([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
sef = dq.asqarray([[0, 0, 0], [0, 0, 1], [0, 0, 0]])

# Tensor product to match system dimensions
b = dq.tensor(b, dq.eye(transmon_trunc))
sge = dq.tensor(dq.eye(aux_trunc), sge)
sef = dq.tensor(dq.eye(aux_trunc), sef)

omega_gf2_values = 1e-3*np.linspace(0.0, 30, 301)
g_aux_values = [30e-3*0.0533*4, 30e-3*0.0533*4*2, 30e-3*0.0533*4*3, 30e-3*0.0533*4*4]

fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)

for g_aux in g_aux_values:
    Gamma_list = []
    for omega_gf2 in omega_gf2_values:

        # System parameters
        g_scaling = 4
        kappa_qubit = 1/10e3  # T1 = 10us
        # 30MHz
        # omega_gf2 = 1.6e-3*5 #25e-3  # 20MHz
        kappa_aux = 3.33e-3  # T1 = 300ns

        # Define Hamiltonian
        H = g_aux * (dq.dag(b) @ sef + b @ dq.dag(sef))
        H += omega_gf2 * (sge @ sef + dq.dag(sef) @ dq.dag(sge))
        H *= 2*jnp.pi

        # Dissipation
        c_ops = [jnp.sqrt(kappa_qubit) * sge, jnp.sqrt(kappa_aux) * b]

        # Initial state
        initial_state = dq.tensor(dq.fock(aux_trunc, 0),
                                dq.fock(transmon_trunc, 0))

        # Simulation
        result = dq.mesolve(H, c_ops, initial_state, t_list)
        final_state = result.states[-1]

        qubit_state_evol = []
        for level in range(transmon_trunc):
            proj = dq.tensor(dq.eye(aux_trunc),
                            dq.fock(transmon_trunc, level) @ dq.dag(dq.fock(transmon_trunc, level)))
            level_pop = (proj @ final_state).trace()
            qubit_state_evol.append(level_pop)
            
        ne = qubit_state_evol[1]
        ng = qubit_state_evol[0]
        Gamma_list.append(np.real(kappa_qubit*ne/ng)*1e3)

    ax.plot(omega_gf2_values*1e3, Gamma_list, label= f"g = {g_aux*1e3:.3f} MHz")

ax.set_title("Qubit population inversion")
ax.set_xlabel("Drive amplitude (MHz)")
ax.set_ylabel("Gamma (MHz)")
ax.legend()


plt.show()