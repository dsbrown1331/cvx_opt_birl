import numpy as np
import mdp_solver
import copy
#bayesian IRL code based on MAP and my own derivation...

def sa_likelihood(state, action, Q_star, eta=1.0):
    """likelihood of state action pair in Bayesian IRL framework
       see Eqn (2) in "Active Learning for Reward Estimation in IRL" """
    num = np.exp(eta * Q_star[state,action])
    denom = np.sum(np.exp(eta * Q_star[state,:]))
    return num / denom


def sa_log_likelihood(state, action, Q_star, eta=1.0):
    log_denom = np.log(np.sum(np.exp(eta * Q_star[state,:])))
    return eta * Q_star[state,action] - log_denom
    
def demo_likelihood(demo, Q_star, eta=1.0):
    return np.prod([sa_likelihood(s,a,Q_star,eta) for s,a in demo])
    
def demo_log_likelihood(demo, Q_star, eta=1.0):
    return np.sum([sa_log_likelihood(s,a,Q_star,eta) for s,a in demo])
    
    
def partial_Q_wrt_R(s,a,i, mdp, pi_star,Wa):
    """partial derivative of Q(s,a) with respect to R_i"""
    #print "T_"+str(a)
    #print mdp.T[:,a,:]
    #print "W"
    #print W
    partial_deriv = mdp.gamma * Wa[s,i]
    if i == s:
        partial_deriv += 1.0 
    #print s,a,"demo"
    #print "partial Q(",s,",",a,") wrt R_",i,"=",partial_deriv
    return partial_deriv

    
def calc_reward_gradient(demo, mdp_r, R, eta=1.0):
    """calculate the gradient of the loglikelihood wrt reward"""
    num_states, num_actions = mdp_r.num_states, mdp_r.num_actions
    #solve mdp with given reward R
    mdp = copy.deepcopy(mdp_r)
    mdp.set_reward(R)
    pi_star, V_star = mdp_solver.policy_iteration(mdp)
    Q_star = mdp_solver.calc_qvals(mdp, pi_star, V_star, mdp.gamma)
    #calculate gradient of R (|s|x1 vector of rewards per state)
    gradient = np.zeros((num_states, 1))
    
    #precompute (I-\gammaT^\pi)^-1
    num_states, num_actions = mdp.num_states, mdp.num_actions
    T_pi = np.array([np.dot(pi_star[x], mdp.T[x]) for x in range(num_states)])
    #print "T_pi"
    #print T_pi
    #TODO check on inverse for better numerical stability         
    Ws = [np.dot(mdp.T[:,a,:], np.linalg.inv(np.eye(mdp.num_states) - mdp.gamma * T_pi)) for a in range(num_actions)]
    
    
    for i in range(num_states): #iterate over reward elements
        r_i_grad = 0
        for s,a in demo: #iterate over demonstrations   
            #print s,a,"demo pair"
            deriv_log_num = eta * partial_Q_wrt_R(s,a,i,mdp,pi_star,Ws[a])
            deriv_log_denom = 1.0/np.sum(np.exp(eta * Q_star[s,:])) \
                    * np.sum([np.exp(eta * Q_star[s,b]) * eta
                    * partial_Q_wrt_R(s,b,i,mdp,pi_star,Ws[b])
                    for b in range(num_actions)])
            #print "deriv_log_num",deriv_log_num
            #print "deriv_log_denom", deriv_log_denom
            r_i_grad += deriv_log_num - deriv_log_denom 
                       
        gradient[i] = r_i_grad
    return gradient

def calc_l1regularized_reward_gradient(demo, mdp_r, R, lam, eta=1.0):
    """calculate subgradient ascent direction to maximize
    P(R|D) - \lambda \|R\|_1"""
    r_grad = calc_reward_gradient(demo, mdp_r, R, eta)
    #
    r_subgrad = r_grad - lam * np.sign(R)
    return r_subgrad
#eta = 1.0
#Q_star = np.array([[1,1,1,1],
#                   [1,1,1,1]])
#demo = [(0,0),(1,1)]
#print sa_likelihood(0,3,Q_star, eta)
#print demo_likelihood(demo, Q_star, eta)
#print demo_log_likelihood(demo, Q_star, eta)
