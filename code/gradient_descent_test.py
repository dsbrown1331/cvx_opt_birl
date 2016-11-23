import numpy as np
import mdp_solver
import gridworld
import util
import birl

#gradient descent on reward
reward = [[0,0,0],
          [0,-1,0],
          [1,-1,0]]
terminals = [6]
gamma = 0.9
simple_world = gridworld.GridWorld(reward, terminals, gamma)
print "reward"
util.print_reward(simple_world)
pi_star, V_star = mdp_solver.policy_iteration(simple_world)
print "optimal policy"
util.print_policy(simple_world, pi_star)
Q_star = mdp_solver.calc_qvals(simple_world, pi_star, V_star, gamma)
print "q-vals"
print Q_star

#give optimal action in each state as demonstration
demo = [(state, np.argmax(Q_star[state,:])) for state in range(simple_world.num_states)]
print demo

#compute the gradient of R_guess
#TODO get an actual guess and update it towards real R 
num_states = simple_world.num_states
num_actions = simple_world.num_actions

print "gradient"
print birl.calc_reward_gradient(demo, simple_world, simple_world.R, eta=1.0)

#test out the log posterior

print "log-likelihood true reward", birl.demo_log_likelihood(demo, Q_star)
reward2 = np.reshape([[0,0,0],
           [0,-1,0],
           [-1,-1,1]],(num_states,1))
#print reward2
#set reward to false values
simple_world.set_reward(reward2)
#calculate new policy
pi_star, V_star = mdp_solver.policy_iteration(simple_world)
print "false reward"
util.print_reward(simple_world)
util.print_policy(simple_world, pi_star)
Q_star = mdp_solver.calc_qvals(simple_world, pi_star, V_star, gamma)
print "log-likelihood false reward", birl.demo_log_likelihood(demo, Q_star)

#try out gradient step to see if likelihood goes up
#I want to try with reward only dependant on states, not actions...
for i in range(10):
    print "----iter",i,"-----"
    print "gradient"
    r_grad = birl.calc_reward_gradient(demo, simple_world, simple_world.R, eta=1.0)
    print r_grad

    #gradient step
    step_size = 0.1
    R_new = simple_world.R + step_size * r_grad
    #print R_new
    simple_world.set_reward(R_new)
    print "updated reward"
    util.print_reward(simple_world)
    pi_star, V_star = mdp_solver.policy_iteration(simple_world)
    print "updated policy"
    util.print_policy(simple_world, pi_star)
    Q_star = mdp_solver.calc_qvals(simple_world, pi_star, V_star, gamma)
    print "updated likelihood reward", birl.demo_log_likelihood(demo, Q_star)


