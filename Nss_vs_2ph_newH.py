import dynamiqs as dq
import numpy as np
import matplotlib.pyplot as plt
import time
import jax.numpy as jnp

# Simulation parameters
res_trunc = 65
aux_trunc = 2
transmon_trunc = 3

t_sim = 7000
timestep = 2
t_list = jnp.arange(0, t_sim, timestep)


# Create destruction operators
a = dq.operators.destroy(res_trunc)
b = dq.operators.destroy(aux_trunc)
c = dq.operators.destroy(transmon_trunc)

# Tensor product to match system dimensions
a = dq.tensor(a, dq.eye(aux_trunc), dq.eye(transmon_trunc))
b = dq.tensor(dq.eye(res_trunc), b, dq.eye(transmon_trunc))
c = dq.tensor(dq.eye(res_trunc), dq.eye(aux_trunc), c)


filename = "saved_state.npz"
fig, ax = plt.subplots(1,2 ,figsize=(9, 4.5), constrained_layout=True)
omega_gf2_list = 1e-3*np.array([72, 76, 80])#np.linspace(62, 80, 21)
Nss_list = []
F_list = []
for omega_gf2 in omega_gf2_list:
    
    # Initial state
    initial_state = dq.tensor(dq.fock(res_trunc, 0),
                                dq.fock(aux_trunc, 0),
                                dq.fock(transmon_trunc, 0))
    
    # System parameters
    wge = 0
    alpha = -200e-3
    waux = alpha
    wgf2 = (wge + waux)/2 

    g_res = 8e-3  # 10MHz
    g_aux = 15e-3  # 30MHz
    kappa_res = 0.2e-3  # T1 = 100us
    kappa_aux = 138e-3  # T1 = 300ns

    # Define Hamiltonian
    H = g_res * (dq.dag(a) @ c + a @ dq.dag(c))
    H += g_aux * (dq.dag(b) @ c + b @ dq.dag(c))
    H += omega_gf2/2 * (c + dq.dag(c))
    H += (wge - wgf2) * dq.dag(a)@a
    H += (waux - wgf2) * dq.dag(b)@b 
    H += (wge - wgf2) * dq.dag(c)@c + alpha/2 * dq.dag(c)@c@(dq.dag(c)@c - dq.eye(*c.dims) )
    H*= 2*jnp.pi

    # Dissipation
    c_ops = [jnp.sqrt(kappa_res) * a, jnp.sqrt(kappa_aux) * b]

    for i in range(2):

        # Simulation
        result = dq.mesolve(H, c_ops, initial_state, t_list, solver = dq.solver.Tsit5(max_steps = 1000000))
        final_state = result.states[-1]
        
        ## Save final state
        np_final_state = np.array(final_state)
        np.savez(filename, state = np_final_state)

        # Update initial state
        np_initial_state = np.load(filename)
        initial_state = dq.asqarray(np_initial_state["state"], dims = (res_trunc, aux_trunc, transmon_trunc))


        
    ## Save final final state
    np_final_state = np.array(final_state)
    np.savez(f"saved_state_{1e3*omega_gf2:.0f}.npz", state = np_final_state)
    
    mean_n = dq.tracemm(dq.dag(a)@a, final_state)
    mean_n_squared = dq.tracemm(dq.dag(a)@a@dq.dag(a)@a, final_state)
    variance_n = mean_n_squared - mean_n**2
    fano_number = variance_n / mean_n
    print(mean_n, fano_number)

    Nss_list.append(mean_n)
    F_list.append(fano_number)


ax[0].plot(omega_gf2_list*1e3, Nss_list, color='blue', alpha=0.7)
ax[0].set_xlabel("2-photon pump (MHz)")
ax[0].set_ylabel("Nss")

ax[1].plot(omega_gf2_list*1e3, F_list, color='blue', alpha=0.7)
ax[1].set_xlabel("2-photon pump (MHz)")
ax[1].set_ylabel("Fano factor")

plt.show()