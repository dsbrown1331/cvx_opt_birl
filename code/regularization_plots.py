#code to plot and analyze the subgradient method with regularization

import numpy as np
import matplotlib.pyplot as plt

#import data from files
num_steps = 500
reg_values = [1.0, 0.5, 0.1, 0.05, 0.01, 0.0]
num_regs = len(reg_values)
reward_diffs = np.zeros((num_steps,num_regs))
reward_norms = np.zeros((num_steps,num_regs))
policy_diffs = np.zeros((num_steps,num_regs))
log_liks = np.zeros((num_steps,num_regs))

cnt = 0
for lam in reg_values: #regularization term

    print "lambda", lam
    fname = "../results/grid_w_reg_lam" + str(lam) + ".txt"
    data = np.loadtxt(fname, delimiter='\t', skiprows=1) #skip header info
    reward_diffs[:,cnt] = data[:num_steps,1]
    reward_norms[:,cnt] = data[:num_steps,2]
    policy_diffs[:,cnt] = data[:num_steps,3]
    log_liks[:,cnt] = data[:num_steps,4]

    cnt += 1

print reward_diffs
print reward_norms
print policy_diffs
print log_liks

plot_data = [reward_diffs, reward_norms, policy_diffs]
ylabels = ["$\|\hat{R} - R^* \|_2$", "$\|\hat{R}\|_1$", "$\|\hat{\pi} - \pi^* \|_2$"]
filenames = ["reward_diff_reg_test", "reward_norm_reg_test", "policy_diff_reg_test"]
legendlocs = ['upper center', 'upper center', 'lower right']

for i in range(len(plot_data)):
    plt.figure(i)
    plt.plot(plot_data[i])
    plt.ylabel(ylabels[i], fontsize=18)
    plt.xlabel("num steps", fontsize=17)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(["$\lambda$ = "+str(r) for r in reg_values],loc=legendlocs[i], fontsize=14)
    plt.savefig("../report/figs/" + filenames[i] + ".png")
plt.show()


