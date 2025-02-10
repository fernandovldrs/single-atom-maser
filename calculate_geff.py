import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.special import jv
import pickle

def calculate_geff(A_flux1, A_flux2, f0, d, p, w_flux_base, N=0, phase = 0):
    # Define functions
    def f_scale(flux, d):
        return np.sqrt(np.abs(np.cos(np.pi * flux) * np.sqrt(1 + d**2 * np.tan(np.pi * flux)**2)))

    def flux_modulation(t, A_flux1, A_flux2, d):
        flux = A_flux1 * np.cos(w_flux_base * t) + A_flux2 * np.cos(w_flux_base * p * t + phase)
        return f_scale(flux, d) * f0


    # Compute Fourier series coefficients of the frequency
    T = 2*np.pi/w_flux_base #t_list[-1] - t_list[0]  # Total period
    T_list = np.linspace(0, T, 500)
    f_time = np.array([flux_modulation(t, A_flux1, A_flux2, d) for t in T_list])
    num_coeffs = 30  # Number of Fourier coefficients to compute
    coeffs = []
    thetas = []
    for k in range(num_coeffs):
        fqk_c = (2 / T) * np.trapz(f_time * np.cos(2 * np.pi * k * T_list / T), T_list)
        fqk_s = (2 / T) * np.trapz(f_time * np.sin(2 * np.pi * k * T_list / T), T_list)
        if k == 0 :
            coeffs.append(fqk_c/2)
            thetas.append(0)
        else:
            fqk = np.sqrt(fqk_c**2 + fqk_s**2)
            thetak = np.arctan2(fqk_c, fqk_s) - np.pi/2
            coeffs.append(fqk)
            thetas.append(thetak)

    # Reconstruct the Fourier series and compute error
    f_reconstructed = np.zeros_like(f_time)
    for k in range(num_coeffs):
        f_reconstructed += coeffs[k] * np.cos(k * w_flux_base * T_list + thetas[k])

    error = np.abs(f_time - f_reconstructed)
    wq_k = [2*np.pi*c for c in coeffs]
    # plt.plot(f_reconstructed)
    # plt.plot(f_time)
    # plt.show()

    # Load selected geff combinations
    with open(f"diophantine_eq_solutions\\selected_combinations_N{N}.pkl", "rb") as f:
        selected_combinations = pickle.load(f)
    n_cutoff = max([max(comb) for comb in selected_combinations])
    k_cutoff = len(selected_combinations[0])

    # Calculate g_eff
    g_eff = 0
    g_eff_factor_list = []
    for comb in selected_combinations:
        g_eff_factor = 1
        for k in range(1, k_cutoff+1):
            J_arg = wq_k[k]/w_flux_base/k
            J = jv(comb[k-1], J_arg)
            J_phase_factor = np.exp(1j*comb[k-1]*thetas[k])
            g_eff_factor *= J*J_phase_factor
        g_eff += g_eff_factor
        g_eff_factor_list.append(g_eff_factor)

    print(f"g_eff = g*{g_eff}")
    print(f"Maximum expansion error = {max(error)*1e3:.3f}MHz")
    # print(n_cutoff, k_cutoff)
    return np.abs(g_eff)


f0 = 8  # in GHz
d = 0.454
p = 3
w_flux_base = 2 * np.pi * 0.275
# Select A_flux1 and A_flux2 values
A_flux1, A_flux2 = 0.1007, 0.1705  # Example value
# A_flux1, A_flux2 = 0.1876, 0.0602  # Example value
calculate_geff(A_flux1, A_flux2, f0, d, p, w_flux_base, N = 2, phase = 0)