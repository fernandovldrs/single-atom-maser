import qutip
import matplotlib.pyplot as plt
import numpy as np

# filename = f'data/laser_threshold_0/bistability/state_402MHz.npz'
filename = f'perturbation2.npz'
loaded_data = np.load(filename)
dims = loaded_data['dims'].tolist()
final_state = qutip.Qobj(loaded_data['data'], dims=dims)

a = qutip.tensor(qutip.destroy(dims[0][0]), 
                    qutip.qeye(dims[0][1]), 
                    qutip.qeye(dims[0][2]))

mean_n = (a.dag()*a* final_state).tr()
mean_n_squared =  ((a.dag()*a)**2 * final_state).tr()
variance_n = mean_n_squared - mean_n**2
fano_number = variance_n / mean_n
print(mean_n, fano_number)

fig, ax = plt.subplots(1, 2, figsize = (7*0.9,5*0.9), constrained_layout=True)

photon_distribution = []
for level in range(dims[0][0]):
    proj = qutip.tensor(qutip.basis(dims[0][0], level)*qutip.basis(dims[0][0], level).dag(),
                        qutip.qeye(dims[0][1]),
                        qutip.qeye(dims[0][2])) 
    level_pop = (proj*final_state).tr()
    photon_distribution.append(level_pop)

p_list = []
for level in range(dims[0][1]): 
    proj = qutip.tensor(qutip.qeye(dims[0][0]),
                        qutip.basis(dims[0][1], level)*qutip.basis(dims[0][1], level).dag(),
                        qutip.qeye(dims[0][2])) 
    level_pop = (proj*final_state).tr()
    p_list.append(level_pop)

proj = qutip.tensor(qutip.qeye(dims[0][0]),
                    qutip.qeye(dims[0][1]),
                    qutip.basis(dims[0][2], 1)*qutip.basis(dims[0][2], 1).dag()) 
p_res = (proj*final_state).tr()

print("Qubit population: ", p_list)
print("RR population: ", p_res)

x = np.linspace(-15, 15, 251)
p = np.linspace(-15, 15, 251)
W_t = qutip.wigner(final_state.ptrace(0), x, p)
extremety = max([np.abs(np.max(W_t)), np.abs(np.min(W_t))])
font_size = 16
border_linewidth = 2
ax[0].bar(range(dims[0][0]), photon_distribution, color='blue', alpha=0.7)
ax[0].set_title("Photon number distribution")
ax[0].set_xlabel("Fock state")
ax[0].set_ylabel("Population")
ax[0].set_ylim([0.0, 1.0])
ax[1].pcolormesh(x, p, W_t.T, cmap = "bwr", vmin = -extremety, vmax = extremety)
ax[1].set_title(r"Wigner function", fontsize=font_size)
ax[1].set_xlabel(r'Re[$\beta$]', fontsize=font_size)
ax[1].set_ylabel(r'Im[$\beta$]', fontsize=font_size)
ax[1].tick_params(axis='both', width=border_linewidth, labelsize=font_size, direction='in', length=8)
plt.show()
