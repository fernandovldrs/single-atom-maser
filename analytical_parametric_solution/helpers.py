import qutip
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.special import jv
import pickle

def f_scale(flux, d):
    return np.sqrt(np.abs(np.cos(np.pi * flux) * np.sqrt(1 + d**2 * np.tan(np.pi * flux)**2)))

def calc_fourier_series(f, T, N, plot = False):

    T_list = np.linspace(0, T, 500)
    f_time = np.array([f(t) for t in T_list])
    coeffs = []
    thetas = []
    for k in range(N):
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

    if plot:
        # Reconstruct the Fourier series and compute error
        f_reconstructed = np.zeros_like(f_time)
        for k in range(N):
            f_reconstructed += coeffs[k] * np.cos(k * 2*np.pi/T * T_list + thetas[k])

        error = np.abs(f_time - f_reconstructed)
        plt.plot(f_reconstructed)
        plt.plot(f_time)
        plt.title(f"Approximation error: {max(error):.3f}")
        plt.show()

    return coeffs, thetas

def calc_fourier_cosine_series(f, T, N, plot = False):
    # Calculate the zeroth Fourier coefficient
    def a0(T):
        integral, _ = quad(f, 0, T)
        return (1/T) * integral
    # Calculate the nth Fourier coefficient
    def an(n, T):
        integral, _ = quad(lambda t, n: f(t) * np.cos(2 * np.pi * n * t / T), 0, T, args=(n,))
        return (2/T) * integral
    # Compute the coefficients
    coeff = [a0(T)] + [an(n, T) for n in range(1, N+1)]

    
    if plot:
        t_values = np.linspace(0, T, 400)
        f_values = f(t_values)
        f_approx = coeff[0] + sum(coeff[n] * np.cos(2 * np.pi * n * t_values / T) for n in range(1, N+1))
        # freq_coeff = [4.62e9, 398e6, - 24.7e6, 2.38e6, -271e3, 33.7e3, - 4.42e3]
        # f_approx2 = freq_coeff[0] + sum(freq_coeff[n] * np.cos(2 * np.pi * n * t_values / T) for n in range(1, N+1))

        plt.figure(figsize=(10, 5))
        plt.plot(t_values, f_values, label='Original function')
        plt.plot(t_values, f_approx, label='Fourier Approximation', linestyle='--')
        # plt.plot(t_values, f_approx2, label='Fourier Approximation2', linestyle='--')
        plt.title('Fourier Series Approximation')
        plt.xlabel('Time t')
        plt.ylabel('f(t)')
        plt.legend()
        plt.show()

    return coeff


def calc_flux_noise(f0, d, p, phi_dc, A_flux1, A_flux2, theta):
    ## Calculates the susceptibility to noise of a qubit
    ## modulated with a two-tone flux pulse.

    phi_ac = np.sqrt(A_flux1**2 + A_flux2**2)
    alpha = np.arcsin(A_flux2/phi_ac)

    # print(phi_ac)
    # print(alpha/2/np.pi)
    # print(theta/2/np.pi)

    # Get the fourier series of the flux curve
    freq_curve = lambda flux: f0*f_scale(flux, d)
    T, N = 1, 10
    freq_coeff = calc_fourier_cosine_series(freq_curve, T, N, plot = False)

    def analytical_f_avg(phi_dc, phi_ac, alpha, theta):
        f_bar = 0
        m_cutoff = 20
        n_cutoff = N
        for m in range(m_cutoff+1):
            for n in range(0, n_cutoff+1):
                s = 1
                s *= np.cos(m*theta)
                s *= freq_coeff[n]
                s *= np.cos(n*2*np.pi*phi_dc + (p + 1)*m*np.pi/2) 
                s *= (2 - (1 if m == 0 else 0) )
                s *= jv(p*m, n*2*np.pi*phi_ac*np.cos(alpha))
                s *= jv(m, n*2*np.pi*phi_ac*np.sin(alpha))
                f_bar += s

        return f_bar

    def dfavg_dphi_dc(phi_dc, phi_ac, alpha, theta):

        f = lambda x: analytical_f_avg(x, phi_ac, alpha, theta)

        def central_difference(f, x, h):
            return (f(x + h) - f(x - h)) / (2 * h)
        
        return central_difference(f, phi_dc, h = 1/5000)

    def dfavg_dphi_ac(phi_dc, phi_ac, alpha, theta):

        f = lambda x: analytical_f_avg(phi_dc, x, alpha, theta)

        def central_difference(f, x, h):
            return (f(x + h) - f(x - h)) / (2 * h)
        
        return central_difference(f, phi_ac, h = 1/5000)

    def decoherence(phi_dc, phi_ac, alpha, theta):
        dphi_dc = dfavg_dphi_dc(phi_dc, phi_ac, alpha, theta)
        dphi_ac = dfavg_dphi_ac(phi_dc, phi_ac, alpha, theta)
        Adc = 33e-6
        Aac = 33e-6 
        return 2*np.pi*3*np.sqrt(Adc**2*dphi_dc**2 + Aac**2*dphi_ac**2)
    
    return decoherence(phi_dc, phi_ac, alpha, theta)

def calculate_geff(A_flux1, A_flux2, f0, d, p, w_flux_base, N=0, phase = 0):
    ## Calculates the effective coupling factor for a given sideband of a parametrically
    ## modulated qubit.

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
    with open(f"analytical_parametric_solution\\diophantine_eq_solutions\\selected_combinations_N{N}.pkl", "rb") as f:
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

    return np.abs(g_eff)
