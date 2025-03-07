import numpy as np
import matplotlib.pyplot as plt
from helper_fns import *
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.special import jv

###########################################################################
##                                                                       ##
## This script calculates the susceptibility to noise of a qubit         ##
## modulated with a two-tone flux pulse.                                 ##
##                                                                       ##
###########################################################################

def calc_flux_noise(f0, d, p, phi_dc, A_flux1, A_flux2, theta):
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

# f0 = 8  # in GHz
# d = 0.454 # SQUID asymmetry
# p = 3
# phi_dc = 0.125
# A_flux1 = 0.7
# A_flux2 = 0.2
# theta = 0.2*2*np.pi
# print(calc_flux_noise(f0, d, p, phi_dc, A_flux1, A_flux2, theta)/1e6/1e-9)