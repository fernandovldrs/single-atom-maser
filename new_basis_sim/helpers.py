
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.special import jv
import pickle
import time

W_COEFF = [
    1, 
    1/2**2, 
    21/2**7, 
    19/2**7, 
    5319/2**15,
    6649/2**15,
    1180581/2**22,
    446287/2**20,
    1489138635/2**31,
    648381403/2**29,
    614557854099/2**38,
    75265839129/2**34,
    637411859250147/2**46,
    86690561488017/2**42,
    405768570324517701/2**53,
    15191635582891041/2**47,
    2497063196283456607731/2**63,
    102281923716042917215/2**57,
    2292687293949773041433127/2**70,
    25544408245062216574759/2**62,
    4971071120163260007203175705/2**78,
    59956026877695226936825271/2**70,
    6299936888270974385982624367587/2**85,
    20465345194746565030172477629/2**75,
    36984324599399309412347250837528543/2**94,
]

ALPHA_COEFF = [
    1,
    9/2**4,
    81/2**7,
    3645/2**12,
    46899/2**15,
    1329129/2**19,
    20321361/2**22,
    2648273373/2**28,
    45579861135/2**31,
    1647988255539/2**35,
    31160327412879/2**38,
    2457206583272505/2**43,
    50387904068904927/2**46,
    2145673984043982897/2**50,
    47368663010124907041/2**53,
    17329540083222030375645/2**60,
    410048712835835979799431/2**63,
    20066784213453521778111375/2**67,
    507447585299180759749453827/2**70,
    53019019946496461235728807475/2**75,
    1429754157181172012054040903645/2**78,
    79571741391885949104006842758911/2**82,
    2283773190022904454409743892590327/2**85,
    540565733415401595950277192471356985/2**91,
    16479511149218202447739080120870460083/2**94,
]

LAMBDA_01_COEFF = [
    1,
    -1/2**3,
    -11/2**8,
    -65/2**11,
    -4203/2**17,
    -40721/2**20,
    -1784885/2**25,
    -21465147/2**28,
    -4455462653/2**35,
    -61698199851/2**38,
    -3623317643901/2**43,
    -56143119646191/2**46,
    -7321743985484303/2**52,
    -125280019793719221/2**55,
    -8984438512815167237/2**60,
    -168544684286400995331/2**63,
    -105741913308715347076701/2**71,
    -2164311753394257835891059/2**74,
    -184798694135089048676718297/2**79,
    -4109869091672376619457585371/2**82,
    -761062061371895548979377743237/2**88,
    -18317012159331390907042783219855/2**91,
    -1831630981593132690479908285273395/2**96,
    -47512263370928552970648689915451821/2**99,
    -20440707519371829420653298425077482201/2**106,
    -569157711742925565406447462105395143103/2**109,
]

LAMBDA_12_COEFF = [
    1,
    -1/2**2,
    -73/2**9,
    -79/2**9,
    -113685/2**19,
    -747533/2**21,
    -175422349/2**28,
    -698471247/2**29,
    -1520876829389/2**39,
    -13668058962903/2**41,
    -4122722770459287/2**48,
    -2534488707574995/2**46,
    -26543348405245135937/2**58,
    -281548290669062665101/2**60,
    -98933257452818263360213/2**67,
    -561603848629069641896937/2**68,
    -3372037991404912212166296765/2**79,
    -40819563311626093062783992331/2**81,
    -16314102788878455728540034311379/2**88,
    -52535388424912627194648863334467/2**88,
    -178610931461508948221684711385383067/2**98,
    -2444937960639526361173164055382471707/2**100,
    -1103567409503040799217165335410059740779/2**107,
    -8017554417550804194373089101907638666069/2**108,
    -30711842188423912661533983529887505235301321/2**118,
    -473069922042437374183190305740304564254754227/2**120,
]

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

def f_scale(flux, d):
    return np.sqrt(np.abs(np.cos(np.pi * flux) * np.sqrt(1 + d**2 * np.tan(np.pi * flux)**2)))

class transmon:
    
    def __init__(self, params, order = 25):

        # Frequency in GHz
        # Flux in units of flux quanta
        
        self.order = order

        self.d = params["d"]
        self.Ec = params["alpha"]
        self.Es = (params["fmax"] + self.Ec)**2/8/self.Ec
        self.Ej1 = self.Es*(1+self.d)/2
        self.Ej2 = self.Es*(1-self.d)/2
        
    def Ej_eff(self, flux):
        varphi = 2*np.pi*flux
        return self.Es*np.sqrt(np.cos(varphi/2)**2 + self.d**2*np.sin(varphi/2)**2)
    
    def varphi_eff(self, flux):
        varphi = 2*np.pi*flux
        return np.arctan(self.d*np.tan(varphi/2))
    
    def xi(self, flux):
        return np.sqrt(2*self.Ec/self.Ej_eff(flux))
    
    def freq(self, flux):
        freq_p = np.sqrt(8 * self.Ec * self.Ej_eff(flux)) # Plasma freq
        return freq_p - self.Ec * sum(coeff * self.xi(flux)**(n) 
                                        for n, coeff in enumerate(W_COEFF[:self.order]))
    
    def alpha(self, flux):
        return self.Ec * sum(coeff * self.xi(flux)**(n) 
                             for n, coeff in enumerate(ALPHA_COEFF[:self.order]))
    
    def lambda01(self, flux):
        return sum(coeff * self.xi(flux)**(n) 
                    for n, coeff in enumerate(LAMBDA_01_COEFF[:self.order]))
    
    def lambda12(self, flux):
        return sum(coeff * self.xi(flux)**(n) 
                    for n, coeff in enumerate(LAMBDA_12_COEFF[:self.order]))

        # Ec = -2*np.pi*alpha
        # EJ = (2*np.pi*f_max + Ec)**2/8/Ec
        # EJ1 = EJ*(1+d)/2
        # EJ2 = EJ*(1-d)/2

def calc_average_transmon(transmon_params, flux_params):

    transm = transmon(transmon_params)

    def flux_modulation(t):
        return sum([A*np.cos(2*np.pi*(freq*t + theta)) 
                    for A, freq, theta in zip(flux_params["As"], flux_params["freqs"], flux_params["phases"])])
        
    T = 1/min(flux_params["freqs"])  # Total period
    T_list = np.linspace(0, T, 500)

    f_avg = np.average([transm.freq(flux_modulation(t)) for t in T_list])
    alpha_avg = np.average([transm.alpha(flux_modulation(t)) for t in T_list])
    lambda01_avg = np.average([transm.lambda01(flux_modulation(t)) for t in T_list])
    lambda12_avg = np.average([transm.lambda12(flux_modulation(t)) for t in T_list])
    xi_avg = np.average([transm.xi(flux_modulation(t)) for t in T_list])

    return f_avg, alpha_avg, lambda01_avg, lambda12_avg, xi_avg
    

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


def calculate_geff(transmon_params, flux_params, Ns=[0], max_nk = 6):
    ## Calculates the effective coupling factor for a given sideband of a parametrically
    ## modulated qubit.

    transm = transmon(transmon_params)

    def flux_modulation(t):
        return sum([A*np.cos(2*np.pi*(freq*t + theta)) 
                    for A, freq, theta in zip(flux_params["As"], flux_params["freqs"], flux_params["phases"])])
        
    wm = 2*np.pi*min(flux_params["freqs"])

    # Compute Fourier series coefficients of the frequency
    T = 2*np.pi/wm  # Total period
    w_t = lambda t: 2*np.pi*transm.freq(flux_modulation(t))
    num_coeffs = 30
    wq_k, theta_k = calc_fourier_series(w_t, T, num_coeffs, plot =  False)

    g_eff_abs_list = []
    for N in Ns:
        # Load selected geff combinations
        with open(f"new_basis_sim\\diophantine_eq_solutions\\selected_combinations_N{N}.pkl", "rb") as f:
            selected_combinations = pickle.load(f)
        # n_cutoff = max([max(comb) for comb in selected_combinations])
        k_cutoff = len(selected_combinations[0])

        # Calculate g_eff
        g_eff = 0
        g_eff_factor_list = []
        
        gamma = sum(wq_k[k]/k/wm*np.sin(theta_k[k]) for k in range(1, k_cutoff+1))
        
        for comb in selected_combinations:
            if not np.all(np.abs(np.array(comb)) < max_nk):
                continue
            g_eff_factor = np.exp(1j*gamma)
            for k in range(1, k_cutoff+1):
                J_arg = wq_k[k]/k/wm
                J = jv(comb[k-1], J_arg)
                J_phase_factor = np.exp(-1j*comb[k-1]*theta_k[k])
                g_eff_factor *= J*J_phase_factor
            g_eff += g_eff_factor
            g_eff_factor_list.append(g_eff_factor)
        g_eff_abs_list.append(np.abs(g_eff))

    return g_eff_abs_list


# flux_mod_params = {
#     "As": (0.332, 0.0),
#     "freqs": (0.275, 3*0.275), # GHz
#     "phases": (0.0, 0.25), # rad/(2pi) 
# }

# transmon_params = {
#     "fmax": 8.0, # GHz
#     "d": 0.454, # SQUID asymmetry
#     "alpha": 0.2, # - anharmonicity, GHz
# } #GF/2 = 7.048551, GE= 7.15175

# print(calculate_geff(flux_mod_params, transmon_params, N = 2))