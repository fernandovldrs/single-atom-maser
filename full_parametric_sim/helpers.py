import qutip
import numpy as np


    
def gaussian_ramp_envelope(t0, pulse_len, ramp_std, ramp_chop, **kwargs):

    def env(t):
        A, B, C, k = t0, ramp_std, pulse_len, ramp_chop
        
        if A < t < k*B + A:
            return np.exp(-(t-(k*B + A))**2/2/B**2)
        elif 2*B + A <= t <= C + 2*B + A:
            return 1.0
        elif C + k*B + A <= t <= C + 2*k*B + A:
            return np.exp(-(t-(C + k*B + A))**2/2/B**2)
        else:
            return 0
        
    return env

class transmon_charge:
    
    def __init__(self, f_max = 6e3, alpha = -200, d = 0, flux = 0, N = 7):
        # This Hamiltonian already assumes the irrotational constraint, so there is no need for
        # explicit mention of the flux derivative.

        varphi = 2*np.pi*flux
        Ec = -2*np.pi*alpha
        EJ = (2*np.pi*f_max + Ec)**2/8/Ec
        # EJ1 = EJ*(1+d)/2
        # EJ2 = EJ*(1-d)/2
        EJ_eff = EJ*np.sqrt(np.cos(varphi/2)**2 + d**2*np.sin(varphi/2)**2)
        varphi_eff = np.arctan(d*np.tan(varphi/2))
        
        ### Hamiltonian in the charge basis
        # Wallraff's group uses N = 15
        H_charge = 4 * Ec * np.diag((np.arange(-N,N+1))**2) 
        H_flux = EJ_eff * 0.5 * (np.diag(-np.ones(2*N), 1)*np.exp(-1j*varphi_eff) +
                                 np.diag(-np.ones(2*N), -1)*np.exp(1j*varphi_eff))
        
        self.H_tr = qutip.Qobj(H_charge + H_flux)
        self.n = qutip.Qobj(np.diag(np.arange(-N, N+1)))
        
    def get_eigenbasis(self):

        ## Transform to eigenbasis
        # E = H_tr.eigenenergies()
        H_tr_eig = self.H_tr.eigenstates()[1]
        return H_tr_eig