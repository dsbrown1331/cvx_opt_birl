import numpy as np
import mdp_solver
import simple_chain
import util
import birl

##test script for running gradient descent for bayesian inverse reinforcement learning
##domain is a simple markov chain (see simple_chain.py) 
##TODO I haven't incorporated a prior so this really is more of a maximum likelihood rather than bayesian irl algorithm

reward = [1,0,0,0,0,1] #true expert reward
terminals = [] #no terminals, you can change this if you want
gamma = 0.9 #discount factor for mdp
chain = simple_chain.SimpleChain(reward, terminals, gamma) #create markov chain
print "expert reward"
print chain.reward
pi_star, V_star = mdp_solver.policy_iteration(chain) #solve for expert policy
print "expert policy"
print pi_star
print "expert value function"
print V_star
Q_star = mdp_solver.calc_qvals(chain, pi_star, V_star, gamma)
print "expert Q-values"
print Q_star

#give optimal action in each (non-terminal) state as demonstration
#we can test giving demonstrations in some but not all states, or even noisy demonstrations to see what happens if we want
demo = [(state, np.argmax(Q_star[state,:])) for state in range(chain.num_states) if state not in terminals]
print "demonstration", demo


####### gradient descent starting from random guess at expert's reward
reward_guess = np.reshape([np.random.randint(-10,10) for _ in range(chain.num_states)],(chain.num_states,1))

#create new mdp with reward_guess as reward
mdp = simple_chain.SimpleChain(reward_guess, terminals, gamma) #create markov chain


num_steps = 200
step_size = 0.1 #we should experiment with step sizes
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
print R_new
pi, V = mdp_solver.policy_iteration(mdp)
print "resulting optimal policy"
print pi
print "policy difference"
print np.linalg.norm(pi_star - pi)
