import numpy as np
import mdp_solver
import gridworld
import util
import birl

##test script for running gradient descent for bayesian inverse reinforcement learning
##domain is a simple grid world (see gridworld.py)
##TODO I haven't incorporated a prior so this really is more of a maximum likelihood rather than bayesian irl algorithm

reward = [[0,0,0,-1,0],[0,-1,0,-1,0],[1,-1,0,0,0]] #true expert reward
terminals = [10] #no terminals, you can change this if you want
gamma = 0.9 #discount factor for mdp
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
reward_guess = np.reshape([np.random.randint(-10,10) for _ in range(grid.num_states)],(grid.rows,grid.cols))

#create new mdp with reward_guess as reward
mdp = gridworld.GridWorld(reward_guess, terminals, gamma) #create markov chain


num_steps = 100
step_size = 1.0 #we should experiment with step sizes
print "----- gradient descent ------"
for step in range(num_steps):
    #calculate optimal policy for current estimate of reward
    pi, V = mdp_solver.policy_iteration(mdp)
    #print "new policy"
    #print pi_star
    #calculate Q values for current estimate of reward
    Q = mdp_solver.calc_qvals(mdp, pi, V, gamma)
    #print "new Qvals"
    #print log-likelihood
    print "log-likelihood posterior", birl.demo_log_likelihood(demo, Q)

    #calculate gradient of posterior wrt reward
    grad = birl.calc_reward_gradient(demo, mdp, mdp.R, eta=1.0)
    #update reward
    R_new = mdp.R + step_size * grad
    #print "new reward"
    #print R_new
    #update mdp with new reward 
    mdp.set_reward(R_new)

print "recovered reward"
util.print_reward(mdp)
pi, V = mdp_solver.policy_iteration(mdp)
print "resulting optimal policy"
util.print_policy(mdp, pi)
print "policy difference"
print np.linalg.norm(pi_star - pi)
