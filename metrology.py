import dynamiqs as dq
import numpy as np
import matplotlib.pyplot as plt
import time
import jax.numpy as jnp

# Simulation parameters
res_trunc = 200
aux_trunc = 2
transmon_trunc = 3

t_sim = 20000
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


# System parameters
g_res = 11e-3*0.04103/2  # 10MHz
g_aux = 30e-3*0.0533/2  # 30MHz
omega_gf2 = 1.6e-3*1.5 #25e-3  # 20MHz
kappa_res = 0.01e-3/1.5  # T1 = 100us
kappa_aux = 3.33e-3  # T1 = 300ns

# System parameters
# g_res = 6.5e-3  # 10MHz
# g_aux = 23.5e-3  # 30MHz
# omega_gf2 = 15e-3 #25e-3  # 20MHz
# kappa_res = 0.31e-3  # T1 = 100us
# kappa_aux = 138e-3  # T1 = 300ns

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


# Simulate the laser disturbance for multiple values of epsilon

# eps = 0.01e-3 # 10kHz
eps_list = np.linspace(0, 0.005e-3, 2)
avg_a_list = []

filename = "saved_state.npz"
np_initial_state = np.load(filename)
print(np_initial_state.files)
initial_state = dq.asqarray(np_initial_state["state"], dims = (res_trunc, aux_trunc, transmon_trunc))

for eps in eps_list:
    # Define Hamiltonian
    H = g_res * (dq.dag(a) @ sge + a @ dq.dag(sge))
    H += g_aux * (dq.dag(b) @ sef + b @ dq.dag(sef))
    H += eps * 1j*(dq.dag(a) - a)
    H += omega_gf2 * (sge @ sef + dq.dag(sef) @ dq.dag(sge))
    H *= 2*jnp.pi

    # Dissipation
    c_ops = [jnp.sqrt(kappa_res) * a, jnp.sqrt(kappa_aux) * b]

    # Simulation
    result = dq.mesolve(H, c_ops, initial_state, t_list)
    avg_a_list.append(dq.tracemm(a, result.states[-1]))

plt.plot(eps_list, avg_a_list)
plt.show()