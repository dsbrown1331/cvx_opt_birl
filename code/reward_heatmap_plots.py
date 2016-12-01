import numpy as np
import matplotlib.pyplot as plt

#plot the recovered reward for lambda = 0.5 for the subgradient method

rmin= -3
rmax = 3

recovered_reward = [[-0.10,  -0.02,  -0.01,  -1.62,  -0.35,  -0.28,  -0.28],
[-0.02,  -1.20,  -0.01,  -1.58,  -0.01,  -1.60,  -0.10],   
[0.01,  -1.37,  -0.02,  -1.46,  -0.01,  -1.74,  -0.47],
[3.32,  -1.34,  -0.27,  -0.25,  -0.02,  -2.22,  -0.87]]  

true_reward = [[0, 0, 0,-1, 0, 0, 0],
          [0,-1, 0,-1, 0,-1, 0],
          [0,-1, 0,-1, 0,-1, 0],
          [1,-1, 0, 0, 0,-1, 0]] #true expert reward

large_reward = [[-1.92,  -1.40,  -1.63,  -10.85,  -2.38,  -2.15,  -2.71],   
[-1.28,  -8.14,  -0.71,  -9.86,  -0.97,  -9.00,  -2.39],   
[-1.57,  -10.27,  -0.64,  -8.61,  -1.12,  -10.99,  -3.02],
[10.03,  -12.45,  -2.08,  -1.92,  -2.10,  -13.83,  -5.34]]   


plt.figure(1)
plt.pcolor(recovered_reward, vmin=rmin, vmax=rmax)
plt.colorbar()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("../report/figs/recovered_reward_lam0.5.png")

plt.figure(2)
plt.pcolor(true_reward, vmin=rmin, vmax=rmax)
plt.colorbar()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("../report/figs/true_reward.png")
#plt.figure(3)
#plt.pcolor(large_reward, vmin=rmin, vmax=rmax)
#plt.colorbar()
plt.show()
