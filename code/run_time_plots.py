#code to plot and analyze the subgradient method with regularization

import numpy as np
import matplotlib.pyplot as plt

#import data from files
num_steps = 10
sizes = range(2,26) #31
num_sizes = len(sizes)
ave_run_time = np.zeros(num_sizes)
x_labels = np.zeros(num_sizes)
cnt = 0
for s in sizes: #size of grid world mdp
    print "size", s
    fname = "../results/runtime_size"+str(s)+".txt"
    reader = open(fname)
    val = reader.read()
    print "run time for 10 steps", val
    ave_run_time[cnt] = float(val)/10.0
    x_labels[cnt] = s * s
    cnt += 1

print ave_run_time


plot_data = [ave_run_time]
ylabels = ["time per gradient calculation (sec)"]
filenames = ["ave_run_time"]
legendlocs = ['lower right']

for i in range(len(plot_data)):
    plt.figure(i)
    plt.plot(x_labels,plot_data[i])
    plt.ylabel(ylabels[i], fontsize=18)
    plt.xlabel("number of states in MDP", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.legend(["$\lambda$ = "+str(r) for r in reg_values],loc=legendlocs[i], fontsize=14)
    plt.savefig("../report/figs/" + filenames[i] + ".png")
plt.show()


