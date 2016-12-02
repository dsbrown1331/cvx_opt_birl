import numpy as np
import mdp_solver
import gridworld
import util
import birl_optimized as birl
import matplotlib.pyplot as plt


##test script for running gradient descent for bayesian inverse reinforcement learning
##domain is a simple grid world (see gridworld.py)
##TODO I haven't incorporated a prior so this really is more of a maximum likelihood rather than bayesian irl algorithm

reward = [[0, 0,0,-1,0, 0,0],
          [0,-1,0,-1,0,-1,0],
          [0,-1,0,-1,0,-1,0],
          [1,-1,0, 0,0,-1,0]] #true expert reward
terminals = [21] #no terminals, you can change this if you want
gamma = 0.95 #discount factor for mdp
grid = gridworld.GridWorld(reward, terminals, gamma) #create grid world
print "expert reward"
util.print_reward(grid)
pi_star, V_star = mdp_solver.policy_iteration(grid) #solve for expert policy
print pi_star
print "expert policy"
util.print_policy(grid, pi_star)
print "expert value function"
util.print_grid(grid, np.reshape(V_star, (grid.rows, grid.cols)))
Q_star = mdp_solver.calc_qvals(grid, pi_star, V_star, gamma)
print "expert Q-values"
print Q_star

#give optimal action in each (non-terminal) state as demonstration
#we can test giving demonstrations in some but not all states, or even noisy demonstrations to see what happens if we want
demo = [(state, np.argmax(Q_star[state,:])) for state in range(grid.num_states) if state not in terminals]
print "demonstration", demo


####### gradient descent starting from random guess at expert's reward
reward_guess = np.reshape([np.random.randint(-1,2) for _ in range(grid.num_states)],(grid.rows,grid.cols))
#reward_guess = np.zeros((grid.rows,grid.cols))
print "reward init", reward_guess

#create new mdp with reward_guess as reward
mdp = gridworld.GridWorld(reward_guess, terminals, gamma) #create markov chain

lam = 0.0 #regularization term
num_steps = 300
save_value1 = np.zeros(num_steps)
save_value2 = save_value1
save_value3 = save_value1
save_value4 = save_value1
save_value5 = save_value1
step_size = 0.1 #we should experiment with step sizes
inc = 0.1
#c = 1.0 #decreasing stepsize
#print "----- gradient descent ------"
for step in range(num_steps):
    print "iter",step
    #calculate optimal policy for current estimate of reward
    pi, V = mdp_solver.policy_iteration(mdp)
    #print "new policy"
    #print pi_star
    #calculate Q values for current estimate of reward
    Q = mdp_solver.calc_qvals(mdp, pi, V, gamma)
    #print "new Qvals"
    #print log-likelihood
    print "log-likelihood posterior", birl.demo_log_likelihood(demo, Q)
    save_value1[step] = birl.demo_log_likelihood(demo, Q)
    #calculate subgradient of posterior wrt reward minus l1 regularization on the reward
    subgrad = birl.calc_l1regularized_reward_gradient(demo, mdp, mdp.R, lam, eta=1.0)
    #update stepsize
    #step_size = c / np.sqrt(step + 1)
    #print "stepsize", step_size
    #update reward
    R_new = mdp.R + step_size * subgrad
    #print "new reward"
    #print R_new
    #update mdp with new reward 
    mdp.set_reward(R_new)

mdp = gridworld.GridWorld(reward_guess, terminals, gamma)
step_size = step_size + inc
for step in range(num_steps):
    print "iter",step
    #calculate optimal policy for current estimate of reward
    pi, V = mdp_solver.policy_iteration(mdp)
    #print "new policy"
    #print pi_star
    #calculate Q values for current estimate of reward
    Q = mdp_solver.calc_qvals(mdp, pi, V, gamma)
    #print "new Qvals"
    #print log-likelihood
    print "log-likelihood posterior", birl.demo_log_likelihood(demo, Q)
    save_value2[step] = birl.demo_log_likelihood(demo, Q)
    #calculate subgradient of posterior wrt reward minus l1 regularization on the reward
    subgrad = birl.calc_l1regularized_reward_gradient(demo, mdp, mdp.R, lam, eta=1.0)
    #update stepsize
    #step_size = c / np.sqrt(step + 1)
    #print "stepsize", step_size
    #update reward
    R_new = mdp.R + step_size * subgrad
    #print "new reward"
    #print R_new
    #update mdp with new reward 
    mdp.set_reward(R_new)

mdp = gridworld.GridWorld(reward_guess, terminals, gamma)
step_size = step_size + inc
for step in range(num_steps):
    print "iter",step
    #calculate optimal policy for current estimate of reward
    pi, V = mdp_solver.policy_iteration(mdp)
    #print "new policy"
    #print pi_star
    #calculate Q values for current estimate of reward
    Q = mdp_solver.calc_qvals(mdp, pi, V, gamma)
    #print "new Qvals"
    #print log-likelihood
    print "log-likelihood posterior", birl.demo_log_likelihood(demo, Q)
    save_value3[step] = birl.demo_log_likelihood(demo, Q)
    #calculate subgradient of posterior wrt reward minus l1 regularization on the reward
    subgrad = birl.calc_l1regularized_reward_gradient(demo, mdp, mdp.R, lam, eta=1.0)
    #update stepsize
    #step_size = c / np.sqrt(step + 1)
    #print "stepsize", step_size
    #update reward
    R_new = mdp.R + step_size * subgrad
    #print "new reward"
    #print R_new
    #update mdp with new reward 
    mdp.set_reward(R_new)

mdp = gridworld.GridWorld(reward_guess, terminals, gamma)
step_size = step_size + inc
for step in range(num_steps):
    print "iter",step
    #calculate optimal policy for current estimate of reward
    pi, V = mdp_solver.policy_iteration(mdp)
    #print "new policy"
    #print pi_star
    #calculate Q values for current estimate of reward
    Q = mdp_solver.calc_qvals(mdp, pi, V, gamma)
    #print "new Qvals"
    #print log-likelihood
    print "log-likelihood posterior", birl.demo_log_likelihood(demo, Q)
    save_value4[step] = birl.demo_log_likelihood(demo, Q)
    #calculate subgradient of posterior wrt reward minus l1 regularization on the reward
    subgrad = birl.calc_l1regularized_reward_gradient(demo, mdp, mdp.R, lam, eta=1.0)
    #update stepsize
    #step_size = c / np.sqrt(step + 1)
    #print "stepsize", step_size
    #update reward
    R_new = mdp.R + step_size * subgrad
    #print "new reward"
    #print R_new
    #update mdp with new rewavrd 
    mdp.set_reward(R_new)

mdp = gridworld.GridWorld(reward_guess, terminals, gamma)
step_size = step_size + inc
for step in range(num_steps):
    print "iter",step
    #calculate optimal policy for current estimate of reward
    pi, V = mdp_solver.policy_iteration(mdp)
    #print "new policy"
    #print pi_star
    #calculate Q values for current estimate of reward
    Q = mdp_solver.calc_qvals(mdp, pi, V, gamma)
    #print "new Qvals"
    #print log-likelihood
    print "log-likelihood posterior", birl.demo_log_likelihood(demo, Q)
    save_value5[step] = birl.demo_log_likelihood(demo, Q)
    #calculate subgradient of posterior wrt reward minus l1 regularization on the reward
    subgrad = birl.calc_l1regularized_reward_gradient(demo, mdp, mdp.R, lam, eta=1.0)
    #update stepsize
    #step_size = c / np.sqrt(step + 1)
    #print "stepsize", step_size
    #update reward
    R_new = mdp.R + step_size * subgrad
    #print "new reward"
    #print R_new
    #update mdp with new reward
    mdp.set_reward(R_new)

plt.clf()
plt.plot(save_value1,label = 'stepsize = 0.1')
#plt.plot(save_value2,label = 'stepsize = 0.2')
#plt.plot(save_value3,label = 'stepsize = 0.3')
#plt.plot(save_value4,label = 'stepsize = 0.4')
plt.plot(save_value5,label = 'stepsize = 0.5')
plt.title('posterior log-likelihood in case of constant stepsize')
plt.legend()
plt.show()



print "recovered reward"
util.print_reward(mdp)
pi, V = mdp_solver.policy_iteration(mdp)
print "resulting optimal policy"
util.print_policy(mdp, pi)
print "policy difference"
print np.linalg.norm(pi_star - pi)
print "l1 norm of reward"
print np.linalg.norm(mdp.R,1)
