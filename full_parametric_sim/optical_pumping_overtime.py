import qutip
import matplotlib.pyplot as plt
import numpy as np
import time
from helpers import transmon_charge
from multiprocessing import Pool
import time

###########################################################################
##                                                                       ##
## This script simulates the dynamics of the qubit as a function of      ##
## time under parametric modulation.                                     ##
## I'm doing this simulation in the charge basis, which considers        ##
## Non-adiabatic transitions and changes to driving parameters.          ##
##                                                                       ##
###########################################################################


# Define frequency curve parameters
f_avg = 7.148  # in GHz
alpha = 0.2

# Define drive properties
N = 7 # Charge operator cutoff
# n = qutip.Qobj(np.diag(np.arange(-N, N+1))) # Charge operator
omega_drive = 0.0127*2*np.pi/1.49376662
freq_drive = - alpha/2 # At the GF/2 point, qubit rotation frame
phi_drive = 0
drive_ramp_std = 5
drive_t0 = 20

# Define readout resonator
rr_trunc = 3
g_res = 0.01658/1.49376662 # Coupling factor in GHz
kappa = 1/30 # Decay

# Define qubit
ref_transmon = transmon_charge(f_max = f_avg, alpha = -alpha, N = N)

# Find the change-of-basis matrix to the reference flux point and define post-COB dimension cutoff
transmon_trunc = 4 # Reduce from 2*N+1 dimensions to transmon_trunc
H_tr = ref_transmon.H_tr
cob_matrix = H_tr.eigenstates()[1]
transmon_Es = H_tr.eigenenergies()
H_offset =  transmon_Es[0]

# Update frequencies given the actual transmon transitions
freq_drive = ((transmon_Es[2] - transmon_Es[0])-2*(transmon_Es[1] - transmon_Es[0]) )/2/2/np.pi
rr_freq = ((transmon_Es[2] - transmon_Es[0])-2*(transmon_Es[1] - transmon_Es[0]) )/2/np.pi

H_tr_diag = H_tr.transform(cob_matrix)
H_tr_diag_offset = H_tr_diag-H_offset
H_transmon = qutip.Qobj(H_tr_diag_offset.tidyup(atol=1e-6)[:transmon_trunc,:transmon_trunc])

meas_basis = [qutip.basis(transmon_trunc, n) for n in range(transmon_trunc)]
proj_list = [qutip.ket2dm(state) for state in meas_basis[:3]] # Projector operators onto g, e and f

n_ch = qutip.Qobj(np.diag(np.arange(-N,N+1))) # charge operator
n_full = n_ch.transform(cob_matrix) # charge operator in eigenbasis
n = n_full[:transmon_trunc,:transmon_trunc]
n_r = np.copy(n) # ladder operator with upper triangule only
n_l = np.copy(n) # ladder operator with lower triangule only
for i in range(transmon_trunc):
    for j in range(transmon_trunc):
        if i>j:
            n_r[i][j] = 0
            n_l[j][i] = 0
n = qutip.Qobj(np.where(np.abs(n) < 1e-6, 0, n))
n_l = qutip.Qobj(np.where(np.abs(n_l) < 1e-6, 0, n_l))
n_r = qutip.Qobj(np.where(np.abs(n_r) < 1e-6, 0, n_r))

# Change to the rotating frame of the qubit
H_rot = qutip.Qobj(np.diag(np.arange(transmon_trunc)))*(transmon_Es[1] - transmon_Es[0])
H_transmon -= H_rot

a = qutip.destroy(rr_trunc)
H_resonator = 2*np.pi*g_res*(qutip.tensor(qutip.Qobj(n_r), a.dag()) + qutip.tensor(qutip.Qobj(n_l), a)) 
H_resonator += 2*np.pi*rr_freq*(qutip.tensor(qutip.qeye(transmon_trunc), a.dag()*a)) 

def H_drive(t, *args):
    drive_len = args[0]["drive_len"]

    A = drive_t0
    B = drive_ramp_std
    C = drive_len

    if A < t < 2*B + A:
        # V = omega_drive*np.cos(2*np.pi*freq_drive*t + phi_drive)
        Vl = omega_drive*np.exp(-1j*2*np.pi*freq_drive*t + phi_drive)
        Vr = omega_drive*np.exp(1j*2*np.pi*freq_drive*t + phi_drive)
        Vl *= np.exp(-(t-(2*B + A))**2/2/B**2)
        Vr *= np.exp(-(t-(2*B + A))**2/2/B**2)
    elif 2*B + A <= t <= C + 2*B + A:
        Vl = omega_drive*np.exp(-1j*2*np.pi*freq_drive*t + phi_drive)
        Vr = omega_drive*np.exp(1j*2*np.pi*freq_drive*t + phi_drive)
    elif C + 2*B + A <= t <= C + 4*B + A:
        Vl = omega_drive*np.exp(-1j*2*np.pi*freq_drive*t + phi_drive)
        Vr = omega_drive*np.exp(1j*2*np.pi*freq_drive*t + phi_drive)
        Vl *= np.exp(-(t-(C + 2*B + A))**2/2/B**2)
        Vr *= np.exp(-(t-(C + 2*B + A))**2/2/B**2)
    else:
        Vl = 0
        Vr = 0
    return Vr*n_r + Vl*n_l

# print(H_drive(0, *{"drive_len":}))
print(n_r)
def H_total(t, *args):
    return qutip.tensor(H_transmon + H_drive(t, *args), qutip.qeye(rr_trunc)) + H_resonator

initial_state = qutip.tensor(meas_basis[0], qutip.basis(rr_trunc, 0))
t_list = np.arange(0, 3064, 1)
start_time = time.time()  # Start timer
c_ops = [np.sqrt(kappa)*qutip.tensor(qutip.qeye(transmon_trunc), qutip.destroy(rr_trunc))]
args = {"drive_len": 3000}
result = qutip.mesolve(H_total, initial_state, t_list, c_ops = c_ops, args = args)

pop0 = [np.real((proj_list[0]*state.ptrace(0)).tr()) for state in result.states]
pop1 = [np.real((proj_list[1]*state.ptrace(0)).tr()) for state in result.states]
pop2 = [np.real((proj_list[2]*state.ptrace(0)).tr()) for state in result.states]
# pop3 = np.real((proj_list[3]*final_state).tr())
# pop4 = np.real((proj_list[4]*final_state).tr())
# rr_pop = np.real((qutip.ket2dm(qutip.basis(rr_trunc, 1))*rr_state).tr())
print(f"Elapsed time: {time.time() - start_time:.6f} seconds")

plt.plot(pop0, label = '0')
plt.plot(pop1, label = '1')
plt.plot(pop2, label = '2')

plt.grid()
plt.ylabel("Ground state population")
plt.xlabel("Time")
plt.legend()
plt.show()