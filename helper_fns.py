import qutip
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

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

def create_qutip_ops(res_trunc, aux_trunc, transmon_trunc):

    # Destruction operators
    a = qutip.destroy(res_trunc)
    b = qutip.destroy(aux_trunc)
    c = qutip.destroy(transmon_trunc) # qubit as ladder
    sge = qutip.Qobj(np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])) # qubit as two transitions
    sef = qutip.Qobj(np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]))

    # Tensor up
    a = qutip.tensor(a, qutip.qeye(aux_trunc), qutip.qeye(transmon_trunc))
    b = qutip.tensor(qutip.qeye(res_trunc), b, qutip.qeye(transmon_trunc))
    c = qutip.tensor(qutip.qeye(res_trunc),  qutip.qeye(aux_trunc), c)
    sge = qutip.tensor(qutip.qeye(res_trunc), qutip.qeye(aux_trunc), sge)
    sef = qutip.tensor(qutip.qeye(res_trunc), qutip.qeye(aux_trunc), sef)

    return a, b, c, sge, sef

def create_qutip_H1(wge, alpha, waux, wgf2, g_res, g_aux, omega_gf2, a, b, c):

    H_1 = g_res*(a.dag()*c + a*c.dag()) 
    H_1 += g_aux * (b.dag() * c + b * c.dag())
    H_1 += omega_gf2/2 * (c + c.dag())
    H_1 += (wge - wgf2) * a.dag()*a
    H_1 += (waux - wgf2) * b.dag()*b
    H_1 += (wge - wgf2) * c.dag()*c + alpha/2 * c.dag()*c*(c.dag()*c - 1 )
    H_1 *= 2*np.pi

    return H_1

def create_qutip_H2(alpha, g_res, g_aux, omega_gf2, a, b, sge, sef):
    alpha = -alpha # different convention 
    H_2 = g_res * (a.dag() * sge + a * sge.dag())
    H_2 += g_aux * np.sqrt(2) * (b.dag() * sef + b * sef.dag())
    H_2 += -omega_gf2**2/np.sqrt(2)/alpha * (sge * sef + sef.dag() * sge.dag())
    H_2 *= 2*np.pi

    return H_2


def create_qutip_initial_state(res_trunc, aux_trunc, transmon_trunc):

    return qutip.tensor(qutip.basis(res_trunc, 0), 
                                qutip.basis(aux_trunc, 0), 
                                qutip.basis(transmon_trunc, 0))

class transmon:
    
    def __init__(self, f_ge = 6e3*2*np.pi, alpha = -200*2*np.pi, g_ef = 8*2*np.pi, g_ge = 11.03*2*np.pi, gamma_res = 0.45, kappa = 50, n_ph = 10,
              dw = 0, f_q = 25*2*np.pi, n_trunc = 4, gamma_tr = 0.1, noRWA = False, w = None):

        
        self.noRWA = noRWA # Still havent figured out what this is 
        self.g_ef = 2*np.pi*g_ef # coupling to aux
        self.g = 2*np.pi*g_ge # coupling to res
        self.gamma_res = gamma_res # res decay
        self.kappa = kappa # aux decay
        self.n_ph = n_ph # res trunc
        self.w = 2*np.pi*w if w else None # drive frequency
        self.dw = 2*np.pi*dw
        self.f_q = 2*np.pi*f_q # drive amplitude
        self.gamma = gamma_tr # transmon decay
        self.n_trunc = n_trunc # transmon trunc
        Ec = -2*np.pi*alpha
        Ej = (2*np.pi*f_ge + Ec)**2/8/Ec

        ### Hamiltonian in the charge basis
        N = 7
        H_charge = 4 * Ec * np.diag((np.arange(-N,N+1))**2) 
        H_flux = 0.5 * Ej * (np.diag(-np.ones(2*N), 1) + np.diag(-np.ones(2*N), -1))
        H_tr = qutip.Qobj(H_charge + H_flux)
        
        E = H_tr.eigenenergies()
        H_tr_eig = H_tr.eigenstates()[1]

        ### If drive frequency not given, make resonant
        if not self.w:
            self.w = (E[2]-E[0])/2
        
        ### Hamiltonian diagonalized in the eigenbasis
        H_tr_diag = H_tr.transform(H_tr_eig)
        H_tr_diag_offset = H_tr_diag-E[0]
        
        n_ch = qutip.Qobj(np.diag(np.arange(-N,N+1))) # charge operator
        n_full = n_ch.transform(H_tr_eig) # charge operator in eigenbasis

        ### Truncate transmon Hamiltonian and charge operator
        H_tr = qutip.Qobj(H_tr_diag_offset.tidyup(atol=1e-3)[:self.n_trunc,:self.n_trunc])
        n = n_full[:self.n_trunc,:self.n_trunc]
        n = np.where(np.abs(n) < 1e-6, 0, n)
        
        self.H_tr = H_tr
        self.n = n
        
    def build_H(self):
        
        H_tr = self.H_tr
        E = H_tr.eigenenergies()
        w = self.w 

        # Interaction picture H_tr
        H0 = qutip.Qobj(np.diag(np.arange(self.n_trunc)))*w # transmon H with freq w
        H_tr_int = H_tr - H0
        H_tr_int = qutip.tensor(qutip.qeye(self.n_ph), H_tr_int, qutip.qeye(2)) # tensor out

        # Interaction picture cavity H
        w_r = E[1]-E[0] # find transmon ge
        H_r = qutip.Qobj(np.diag(np.arange(self.n_ph)))*w_r # cavity H with same w as transmon ge
        H0_res = qutip.Qobj(np.diag(np.arange(self.n_ph)))*w # cavity H with freq w
        H_r_int = H_r-H0_res # interaction picture
        H_r_int = qutip.tensor(H_r_int, qutip.qeye(self.n_trunc), qutip.qeye(2)) # tensor out

        # Interaction picture aux H
        w_ef = E[2]-E[1] # find transmon wef
        H_aux = qutip.Qobj(np.diag(np.arange(2)))*w_ef # aux H with same w as transmon ef
        H0_aux = qutip.Qobj(np.diag(np.arange(2)))*w # aux H with freq w
        H_aux_int = H_aux-H0_aux # interaction picture
        H_aux_int = qutip.tensor(qutip.qeye(self.n_ph), qutip.qeye(self.n_trunc), H_aux_int) # tensor out

        ### Create ladder operators from the charge operator
        n = self.n
        n_r = np.copy(n) # ladder operator with upper triangule only
        n_l = np.copy(n) # ladder operator with lower triangule only
        for i in range(self.n_trunc):
            for j in range(self.n_trunc):
                if i>j:
                    n_r[i][j] = 0
                    n_l[j][i] = 0

        # This is the approximate way of defining the ladder operators of the transmon.
        # It is not accurate enough to reproduce the laser threshold reliably.
        # ### Create ladder operators from the charge operator
        # n_r = destroy(self.n_trunc)
        # n_l = create(self.n_trunc)
        # n = n_r + n_l

        ### Create qubit-cavity interaction
        a = qutip.destroy(self.n_ph)
        ac = qutip.create(self.n_ph)
        H_int_res = self.g*(qutip.tensor(ac, qutip.Qobj(n_r), qutip.qeye(2)) + qutip.tensor(a, qutip.Qobj(n_l), qutip.qeye(2))) 

        ### Create qubit-RR interaction
        a_2 = qutip.destroy(2)
        ac_2 = qutip.create(2)
        H_int_ef = self.g_ef*(qutip.tensor(qutip.qeye(self.n_ph), qutip.Qobj(n_r), ac_2) + qutip.tensor(qutip.qeye(self.n_ph), qutip.Qobj(n_l), a_2))

        ### Create drive hamiltonian from charge operator
        f_q = self.f_q
        Hd = qutip.Qobj(n*f_q/2)
        Hd = qutip.tensor(qutip.qeye(self.n_ph), Hd, qutip.qeye(2))
        H = H_tr_int + H_r_int + H_aux_int + H_int_res + H_int_ef + Hd

        return H
    
    def build_C(self):
        
        n = self.n
        
        if self.n_trunc == 4:
            C_tr = qutip.Qobj([[0, 1, 0, 0],[0, 0, np.abs(n[1][2]/n[0][1]), 0],[0, 0, 0, np.abs(n[2][3]/n[1][2])], [0, 0, 0, 0]])
        else:
            C_tr = qutip.Qobj([[0, 1, 0],[0, 0, np.abs(n[1][2]/n[0][1])],[0, 0, 0]])
            
        gamma = self.gamma
        C_dis_tr = qutip.tensor(qutip.qeye(self.n_ph), np.sqrt(gamma)*C_tr, qutip.qeye(2))
        C_dis_ef = qutip.tensor(qutip.qeye(self.n_ph), qutip.qeye(self.n_trunc), np.sqrt(self.kappa)*qutip.destroy(2))
        C_dis_res = qutip.tensor(np.sqrt(self.gamma_res)*qutip.destroy(self.n_ph), qutip.qeye(self.n_trunc), qutip.qeye(2))
        
        import scipy.sparse as sp
        C_dis_tr = qutip.Qobj(sp.csr_matrix(C_dis_tr.full(), dtype=complex))
        C_dis_ef = qutip.Qobj(sp.csr_matrix(C_dis_ef.full(), dtype=complex))
        C_dis_res = qutip.Qobj(sp.csr_matrix(C_dis_res.full(), dtype=complex))
        return [C_dis_tr, C_dis_ef, C_dis_res]
    