import numpy as np
from scipy.special import jv
import pickle
from helper_fns import *

def calculate_geff(A_flux1, A_flux2, f0, d, p, w_flux_base, N=0, phase = 0):

    def flux_modulation(t, A_flux1, A_flux2, d):
        flux = A_flux1 * np.cos(w_flux_base * t) + A_flux2 * np.cos(w_flux_base * p * t + phase)
        return f_scale(flux, d) * f0

    # Compute Fourier series coefficients of the frequency
    T = 2*np.pi/w_flux_base  # Total period
    f = lambda t: flux_modulation(t, A_flux1, A_flux2, d)
    num_coeffs = 30
    coeffs, thetas = calc_fourier_series(f, T, num_coeffs, plot = False)
    wq_k = [2*np.pi*c for c in coeffs]

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
    return np.abs(g_eff)


# f0 = 8  # in GHz
# d = 0.454
# p = 3
# w_flux_base = 2 * np.pi * 0.275
# # Select A_flux1 and A_flux2 values
# A_flux1, A_flux2 = 0.1007, 0.1705  # Example value
# # A_flux1, A_flux2 = 0.1876, 0.0602  # Example value
# calculate_geff(A_flux1, A_flux2, f0, d, p, w_flux_base, N = 2, phase = 0)