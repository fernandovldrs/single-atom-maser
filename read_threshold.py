import qutip
import matplotlib.pyplot as plt
import numpy as np
import time
from helper_fns import *
import scipy.sparse as sp
import os
from scipy.optimize import curve_fit

# Simulation parameters
res_trunc_start = 30
transmon_trunc = 4
aux_trunc = 2

# Characteristic function plot range
beta_range = 4 
beta_step = 0.05
beta_list = np.arange(-beta_range, beta_range + beta_step/2, beta_step)

folder = "sol_45/"
# List all states in the directory
files = os.listdir(folder)
omega_gf2_list = []
for file in files:
    omega_gf2 = int(file.split("_")[1].split(".")[0])
    omega_gf2_list.append(omega_gf2)

omega_gf2_list.sort()

Nss_list = []
F_list = []
char_function_list = []
res_trunc = res_trunc_start
for omega_gf2 in omega_gf2_list:
    filename = folder + f'state_{omega_gf2:.0f}.npz'
    loaded_data = np.load(filename)
    final_state = qutip.Qobj(loaded_data['data'], dims=loaded_data['dims'].tolist())
    found_dims = False
    while not found_dims:
        try:
            a = qutip.tensor(qutip.destroy(res_trunc), qutip.qeye(transmon_trunc), qutip.qeye(aux_trunc))
            mean_n = (a.dag()*a* final_state).tr()
            mean_n_squared =  ((a.dag()*a)**2 * final_state).tr()
            variance_n = mean_n_squared - mean_n**2
            fano_number = variance_n / mean_n
            print("Nss: ", np.real(mean_n), "Fano: ", np.real(fano_number), "Trunc: ", res_trunc)

            Nss_list.append(mean_n)
            F_list.append(fano_number)
            found_dims = True
            
            char_function = []
            res_state = final_state.ptrace(0)
            for beta in beta_list:
                    D_beta = displace(res_state.dims[0][0], beta)  # Displacement operator
                    char_function_point = (res_state * D_beta).tr() #* np.exp(0.5 * np.abs(beta)**2)
                    char_function.append(np.real(char_function_point))
            char_function_list.append(char_function)
        except:
            res_trunc += 5

plt.plot(omega_gf2_list, Nss_list)
plt.show()
plt.plot(omega_gf2_list, F_list)
plt.show()

for i, char_function in enumerate([char_function_list]):
    plt.plot(beta_list, char_function, label = i)

# #plot coherent state
# alpha = np.sqrt(35)*1j
# char_function = []
# state = ket2dm(coherent(70, alpha))
# diagonal_elements = np.diag(state.full())
# diagonal_matrix = np.diag(diagonal_elements)
# state = qutip.Qobj(diagonal_matrix)

# # state = ket2dm(basis(90, 41))
# for beta in beta_list:
#     D_beta = displace(state.dims[0][0], beta)  # Displacement operator
#     char_function_point = (state * D_beta).tr() #* np.exp(0.5 * np.abs(beta)**2)
#     char_function.append(np.real(char_function_point))
        
# plt.plot(beta_list, char_function)

plt.legend()
plt.show()
