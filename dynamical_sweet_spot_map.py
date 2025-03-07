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


f0 = 8  # in GHz
d = 0.454 # SQUID asymmetry
p = 3
phi_dc = 0.0
phi_ac = 0.72801

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

alpha_list = np.linspace(0, 2*np.pi*0.25, 100)
theta_list = np.linspace(0, 2*np.pi*0.5, 100)
X, Y = np.meshgrid(alpha_list, theta_list)
Z = decoherence(phi_dc, phi_ac, X, Y)/1e6
Z /= 1e-9 # to match scale in the paper

# Create the plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(X/2/np.pi, Y/2/np.pi, Z, 50, cmap='terrain',vmin=0, vmax=2.75)  # 20 levels and a colormap
plt.colorbar(contour)  # Add a colorbar to a plot
plt.contour(X/2/np.pi, Y/2/np.pi, Z, 50, colors='white', linewidths=0.2)  # Level contours in black
plt.title('2D Color Map with Level Contour')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()
