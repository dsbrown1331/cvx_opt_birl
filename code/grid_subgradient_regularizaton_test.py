import numpy as np
import mdp_solver
import gridworld
import util
import birl_optimized as birl

##test script for running gradient descent for bayesian inverse reinforcement learning
##domain is a simple grid world (see gridworld.py)
##TODO I haven't incorporated a prior so this really is more of a maximum likelihood rather than bayesian irl algorithm

reward = [[0, 0, 0,-1, 0, 0, 0],
          [0,-1, 0,-1, 0,-1, 0],
          [0,-1, 0,-1, 0,-1, 0],
          [1,-1, 0, 0, 0,-1, 0]] #true expert reward
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
reward_guess = np.reshape([np.random.randint(-10,11) for _ in range(grid.num_states)],(grid.rows,grid.cols))
#reward_guess = np.zeros((grid.rows,grid.cols))
#reward_guess = np.ones((grid.rows,grid.cols))
print "reward init", reward_guess

#create new mdp with reward_guess as reward
mdp = gridworld.GridWorld(reward_guess, terminals, gamma) #create markov chain




for lam in [1.0, 0.5, 0.1, 0.05, 0.01, 0.0]: #regularization term
    print "lambda", lam
    f = open("../results/grid_w_reg_lam" + str(lam) + ".txt","w")
    f.write("iter\treward-diff\treward-norm\tpolicy-diff\tlqog-lik\n")
    num_steps = 500
    #step_size = 0.1 #we should experiment with step sizes
    c = 0.5 #decreasing stepsize
    print "----- gradient descent ------"
    for step in range(num_steps):
        #print "iter",step
        #calculate optimal policy for current estimate of reward
        pi, V = mdp_solver.policy_iteration(mdp)
        #print "new policy"
        #print pi_star
        #calculate Q values for current estimate of reward
        Q = mdp_solver.calc_qvals(mdp, pi, V, gamma)
        #print "new Qvals"
        #print log-likelihood
        log_lik = birl.demo_log_likelihood(demo, Q)
        #print "log-likelihood posterior", log_lik

        #calculate subgradient of posterior wrt reward minus l1 regularization on the reward
        subgrad = birl.calc_l1regularized_reward_gradient(demo, mdp, mdp.R, lam, eta=1.0)
        #update stepsize
        step_size = c / np.sqrt(step + 2)
        #print "stepsize", step_size
        #update reward
        R_new = mdp.R + step_size * subgrad
        #print "new reward"
        #print R_new
        #update mdp with new reward 
        mdp.set_reward(R_new)
        reward_diff = np.linalg.norm(grid.R - R_new)
        reward_norm = np.linalg.norm(R_new,1)
        policy_diff = np.linalg.norm(pi_star - pi)

        

        f.write("%d\t%f\t%f\t%f\t%f\n" %(step, reward_diff, reward_norm, policy_diff, log_lik))
    f.close()

    print "recovered reward"
    util.print_reward(mdp)
    pi, V = mdp_solver.policy_iteration(mdp)
    print "resulting optimal policy"
    util.print_policy(mdp, pi)
    print "policy difference"
    print np.linalg.norm(pi_star - pi)
    print "l1 norm of reward"
    print np.linalg.norm(mdp.R,1)
