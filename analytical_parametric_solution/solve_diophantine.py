import itertools
import pickle

###########################################################################
##                                                                       ##
## This script finds solutions to a given diophantine equation in the    ##
## integer domain with given cutoffs. This is used to calculate          ##
## the effective coupling factor between qubit and resonator during      ##
## parametric modulation.                                                ##
##                                                                       ##
###########################################################################

k_cutoff = 6
n_cutoff = 6
N = -4
# test
# Generate all possible combinations that solve the Diophantine eq.
values = range(-n_cutoff, n_cutoff + 1)
selected_combinations = []
for comb in itertools.product(values, repeat=k_cutoff):
    if sum([comb[k-1]*k for k in range(1, k_cutoff+1)]) == N:
        selected_combinations.append(comb)

# Save to file
with open("analytical_parametric_solution\\diophantine_eq_solutions\\selected_combinations.pkl", "wb") as f:
    pickle.dump(selected_combinations, f)

print("Selected combinations saved.")
