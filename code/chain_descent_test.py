import numpy as np
import mdp_solver
import simple_chain
import util
import birl

#gradient descent on reward
reward = [0,1]
terminals = []
gamma = 0.9
simple_world = simple_chain.SimpleChain(reward, terminals, gamma)
print "reward"
print simple_world.reward
pi_star, V_star = mdp_solver.policy_iteration(simple_world)
print "optimal policy"
print pi_star
print "values"
print V_star
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
reward2 = np.reshape([1,-1],(num_states,1))


print "new reward"
print reward2
#set reward to false values
simple_world.set_reward(reward2)

num_steps = 1000
step_size = 0.1 #any step seems to work 
print "----- gradient descent starting from wrong reward ------"
for step in range(num_steps):
    print "--step",step,"---"
    #calculate new policy
    pi_star, V_star = mdp_solver.policy_iteration(simple_world)
    print "new policy"
    print pi_star
    print "new values"
    print V_star
    print "false reward"
    Q_star = mdp_solver.calc_qvals(simple_world, pi_star, V_star, gamma)
    print "new Qvals"
    print Q_star
    print "log-likelihood false reward", birl.demo_log_likelihood(demo, Q_star)

    print "gradient"
    grad = birl.calc_reward_gradient(demo, simple_world, simple_world.R, eta=1.0)
    print grad
    R_new = simple_world.R + step_size * grad
    print "new reward"
    print R_new
    simple_world.set_reward(R_new)


