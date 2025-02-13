import dynamiqs as dq
import numpy as np
import matplotlib.pyplot as plt
import time
import jax.numpy as jnp

# Simulation parameters
aux_trunc = 3
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

# System parameters
g_scaling = 4
kappa_qubit = 1/10e3  # T1 = 10us
g_aux = 30e-3*0.0533*g_scaling  # 30MHz
omega_gf2 = 1.6e-3*5 #25e-3  # 20MHz
kappa_aux = 3.33e-3  # T1 = 300ns

# Define Hamiltonian
H = g_aux * (dq.dag(b) @ sef + b @ dq.dag(sef))
H += omega_gf2 * (sge @ sef + dq.dag(sef) @ dq.dag(sge))
H *= 2*jnp.pi

# Calculate effective qubit pumping rate
## Calculate effective qubit loss to resonator
if kappa_aux > 4*g_aux: # overdamped regime 
    print("RR is overdamped")
    kappa_aux_eff = 0.5*(kappa_aux - np.sqrt(kappa_aux**2 - 16*g_aux**2))
else: # underdamped regime
    print("RR is underdamped")
    kappa_aux_eff = 0.5*(kappa_aux)
## Calculate effective qubit incoherent excitation
if kappa_aux_eff > 2*omega_gf2: # overdamped regime 
    print("Qubit EF is overdamped")
    gamma = 0.5*(kappa_aux_eff - np.sqrt(kappa_aux_eff**2 - 4*omega_gf2**2))
else: # underdamped regime
    print("Qubit EF is underdamped")
    gamma = 0.5*(kappa_aux_eff)

print(f"Effective qubit EF loss rate kappa_aux_eff = {kappa_aux_eff*1e3:.3f} MHz")
print(f"Effective qubit pumping rate Gamma = {gamma*1e3:.3f} MHz")


# Dissipation
c_ops = [jnp.sqrt(kappa_qubit) * sge, jnp.sqrt(kappa_aux) * b]

# Initial state
initial_state = dq.tensor(dq.fock(aux_trunc, 0),
                          dq.fock(transmon_trunc, 0))

# Simulation
result = dq.mesolve(H, c_ops, initial_state, t_list)


# Plot results
fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)

# qubit_g_evol = []
# qubit_e_evol = []
qubit_state_evol = [[], [], []]
aux_state_evol = [[], [], []]
for state in result.states:
    for level in range(transmon_trunc):
        proj = dq.tensor(dq.eye(aux_trunc),
                        dq.fock(transmon_trunc, level) @ dq.dag(dq.fock(transmon_trunc, level)))
        level_pop = (proj @ state).trace()
        qubit_state_evol[level].append(level_pop)

    for level in range(aux_trunc):
        proj = dq.tensor(dq.fock(aux_trunc, level) @ dq.dag(dq.fock(aux_trunc, level)),
                         dq.eye(transmon_trunc),)
        level_pop = (proj @ state).trace()
        aux_state_evol[level].append(level_pop)

for level in range(transmon_trunc):
    ax[0].plot(t_list, qubit_state_evol[level], label = level)
for level in range(aux_trunc):
    ax[1].plot(t_list, aux_state_evol[level], label = level)

if np.any(np.array(aux_state_evol[2]) > 0.01):
    print("Resonator Fock 2 population might be too high")

ax[0].set_title("Qubit state evolution")
ax[0].set_xlabel("Time (ns)")
ax[0].set_ylabel("Population")
ax[0].set_ylim([-0.05, 1.05])
ax[0].legend()
ax[1].legend()

ax[1].set_title("RR state evolution")
ax[1].set_xlabel("Time (ns)")
ax[1].set_ylabel("Population")
ax[1].set_ylim([-0.05, 1.05])
ax[1].legend()

ne = qubit_state_evol[1][-1]
ng = qubit_state_evol[0][-1]
print("If the populations converged: Gamma = kappa_qubit*ne/ng")
print(f"Gamma = {np.real(kappa_qubit*ne/ng)*1e3} MHz")

plt.show()