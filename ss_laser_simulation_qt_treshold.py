import qutip
from qutip.solver import Options
import matplotlib.pyplot as plt
import numpy as np
from qutip import wigner
import time

Nss_list = []
F_list = []

# Simulation parameters
res_trunc = 15
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

# Define ladder operators for the transmon
sge = qutip.Qobj(np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
sef = qutip.Qobj(np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]))
sge = qutip.tensor(qutip.qeye(res_trunc), qutip.qeye(aux_trunc), sge)
sef = qutip.tensor(qutip.qeye(res_trunc), qutip.qeye(aux_trunc), sef)


# Simulation
initial_state = qutip.tensor(qutip.basis(res_trunc, 0), 
                                qutip.basis(aux_trunc, 0), 
                                qutip.basis(transmon_trunc, 0))

fig, ax = plt.subplots(1,2 ,figsize=(9, 4.5), constrained_layout=True)
omega_gf2_list = np.arange(2, 32, 2)
for omega_gf2 in omega_gf2_list:
    
    print("\nOmega = ", omega_gf2)
    # System parameters
    wge = 0
    alpha = -200
    waux = alpha
    wgf2 = (wge + waux)/2 
    g_res = 11 # 10MHz
    g_aux = 8  # 30MHz
    # omega_gf2 = 25  # 20MHz
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

    # Losses
    c_ops = [np.sqrt(kappa_res) * a, np.sqrt(kappa_aux) * b]

    L_1 = qutip.liouvillian(H_1, c_ops)

    t1 = time.time()
    final_state = qutip.steadystate(L_1, method = 'power', use_rcm = True)
    t2 = time.time()
    print(t2-t1)

    ## Save final states
    np_final_state = np.array(final_state)
    np.savez(f"saved_state_{omega_gf2:.0f}.npz", state = np_final_state)
    
    ## Print warnings
    projN = qutip.tensor(qutip.basis(res_trunc, res_trunc-1)*qutip.basis(res_trunc, res_trunc-1).dag(), 
                        qutip.qeye(aux_trunc),
                        qutip.qeye(transmon_trunc)) 
    if np.real((projN*final_state).tr()) > 0.001 or np.real((projN*final_state).tr()) > 0.001:
        print("Increase truncation!")

    mean_n_1 = (a.dag()*a*final_state).tr()
    mean_n_squared_1 = qutip.expect((a.dag()*a)**2, final_state)
    variance_n_1 = mean_n_squared_1 - mean_n_1**2
    fano_number_1 = variance_n_1 / mean_n_1
    print("H1: ", mean_n_1, fano_number_1)


# Simulation parameters
res_trunc = 25
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

# Define ladder operators for the transmon
sge = qutip.Qobj(np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
sef = qutip.Qobj(np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]))
sge = qutip.tensor(qutip.qeye(res_trunc), qutip.qeye(aux_trunc), sge)
sef = qutip.tensor(qutip.qeye(res_trunc), qutip.qeye(aux_trunc), sef)


# Simulation
initial_state = qutip.tensor(qutip.basis(res_trunc, 0), 
                                qutip.basis(aux_trunc, 0), 
                                qutip.basis(transmon_trunc, 0))

fig, ax = plt.subplots(1,2 ,figsize=(9, 4.5), constrained_layout=True)
omega_gf2_list = np.arange(32, 44, 2)
for omega_gf2 in omega_gf2_list:
    
    print("\nOmega = ", omega_gf2)
    # System parameters
    wge = 0
    alpha = -200
    waux = alpha
    wgf2 = (wge + waux)/2 
    g_res = 11 # 10MHz
    g_aux = 8  # 30MHz
    # omega_gf2 = 25  # 20MHz
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

    # Losses
    c_ops = [np.sqrt(kappa_res) * a, np.sqrt(kappa_aux) * b]

    L_1 = qutip.liouvillian(H_1, c_ops)

    t1 = time.time()
    final_state = qutip.steadystate(L_1, method = 'power', use_rcm = True)
    t2 = time.time()
    print(t2-t1)

    ## Save final states
    np_final_state = np.array(final_state)
    np.savez(f"saved_state_{omega_gf2:.0f}.npz", state = np_final_state)
    
    ## Print warnings
    projN = qutip.tensor(qutip.basis(res_trunc, res_trunc-1)*qutip.basis(res_trunc, res_trunc-1).dag(), 
                        qutip.qeye(aux_trunc),
                        qutip.qeye(transmon_trunc)) 
    if np.real((projN*final_state).tr()) > 0.001 or np.real((projN*final_state).tr()) > 0.001:
        print("Increase truncation!")

    mean_n_1 = (a.dag()*a*final_state).tr()
    mean_n_squared_1 = qutip.expect((a.dag()*a)**2, final_state)
    variance_n_1 = mean_n_squared_1 - mean_n_1**2
    fano_number_1 = variance_n_1 / mean_n_1
    print("H1: ", mean_n_1, fano_number_1)



# Simulation parameters
res_trunc = 35
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

# Define ladder operators for the transmon
sge = qutip.Qobj(np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
sef = qutip.Qobj(np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]))
sge = qutip.tensor(qutip.qeye(res_trunc), qutip.qeye(aux_trunc), sge)
sef = qutip.tensor(qutip.qeye(res_trunc), qutip.qeye(aux_trunc), sef)


# Simulation
initial_state = qutip.tensor(qutip.basis(res_trunc, 0), 
                                qutip.basis(aux_trunc, 0), 
                                qutip.basis(transmon_trunc, 0))

fig, ax = plt.subplots(1,2 ,figsize=(9, 4.5), constrained_layout=True)
omega_gf2_list = np.arange(44, 60, 2)
for omega_gf2 in omega_gf2_list:
    
    print("\nOmega = ", omega_gf2)
    # System parameters
    wge = 0
    alpha = -200
    waux = alpha
    wgf2 = (wge + waux)/2 
    g_res = 11 # 10MHz
    g_aux = 8  # 30MHz
    # omega_gf2 = 25  # 20MHz
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

    # Losses
    c_ops = [np.sqrt(kappa_res) * a, np.sqrt(kappa_aux) * b]

    L_1 = qutip.liouvillian(H_1, c_ops)

    t1 = time.time()
    final_state = qutip.steadystate(L_1, method = 'power', use_rcm = True)
    t2 = time.time()
    print(t2-t1)

    ## Save final states
    np_final_state = np.array(final_state)
    np.savez(f"saved_state_{omega_gf2:.0f}.npz", state = np_final_state)
    
    ## Print warnings
    projN = qutip.tensor(qutip.basis(res_trunc, res_trunc-1)*qutip.basis(res_trunc, res_trunc-1).dag(), 
                        qutip.qeye(aux_trunc),
                        qutip.qeye(transmon_trunc)) 
    if np.real((projN*final_state).tr()) > 0.001 or np.real((projN*final_state).tr()) > 0.001:
        print("Increase truncation!")

    mean_n_1 = (a.dag()*a*final_state).tr()
    mean_n_squared_1 = qutip.expect((a.dag()*a)**2, final_state)
    variance_n_1 = mean_n_squared_1 - mean_n_1**2
    fano_number_1 = variance_n_1 / mean_n_1
    print("H1: ", mean_n_1, fano_number_1)






# Simulation parameters
res_trunc = 55
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

# Define ladder operators for the transmon
sge = qutip.Qobj(np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
sef = qutip.Qobj(np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]))
sge = qutip.tensor(qutip.qeye(res_trunc), qutip.qeye(aux_trunc), sge)
sef = qutip.tensor(qutip.qeye(res_trunc), qutip.qeye(aux_trunc), sef)


# Simulation
initial_state = qutip.tensor(qutip.basis(res_trunc, 0), 
                                qutip.basis(aux_trunc, 0), 
                                qutip.basis(transmon_trunc, 0))

fig, ax = plt.subplots(1,2 ,figsize=(9, 4.5), constrained_layout=True)
omega_gf2_list = np.arange(60, 82, 2)
for omega_gf2 in omega_gf2_list:
    
    print("\nOmega = ", omega_gf2)
    # System parameters
    wge = 0
    alpha = -200
    waux = alpha
    wgf2 = (wge + waux)/2 
    g_res = 11 # 10MHz
    g_aux = 8  # 30MHz
    # omega_gf2 = 25  # 20MHz
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

    # Losses
    c_ops = [np.sqrt(kappa_res) * a, np.sqrt(kappa_aux) * b]

    L_1 = qutip.liouvillian(H_1, c_ops)

    t1 = time.time()
    final_state = qutip.steadystate(L_1, method = 'power', use_rcm = True)
    t2 = time.time()
    print(t2-t1)

    ## Save final states
    np_final_state = np.array(final_state)
    np.savez(f"saved_state_{omega_gf2:.0f}.npz", state = np_final_state)
    
    ## Print warnings
    projN = qutip.tensor(qutip.basis(res_trunc, res_trunc-1)*qutip.basis(res_trunc, res_trunc-1).dag(), 
                        qutip.qeye(aux_trunc),
                        qutip.qeye(transmon_trunc)) 
    if np.real((projN*final_state).tr()) > 0.001 or np.real((projN*final_state).tr()) > 0.001:
        print("Increase truncation!")

    mean_n_1 = (a.dag()*a*final_state).tr()
    mean_n_squared_1 = qutip.expect((a.dag()*a)**2, final_state)
    variance_n_1 = mean_n_squared_1 - mean_n_1**2
    fano_number_1 = variance_n_1 / mean_n_1
    print("H1: ", mean_n_1, fano_number_1)


