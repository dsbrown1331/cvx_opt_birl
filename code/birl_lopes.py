import numpy as np
import mdp_solver
import copy
#bayesian IRL code

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
    
def partial_Lsu_partial_Rxa(s, u, x, a, mdp, Q_star, pi_star, eta):
    """calculate the partial derivative of likelihood of (s,u) pair 
       w.r.t. reward at state x taking action a"""
       
    kron_delta_xa = 0.0
    kron_delta_x = 0.0       
    if x == s:
        kron_delta_x = 1.0
        if a == u:
            kron_delta_xa = 1.0
       
       
    delLsu_delQxa = eta * sa_likelihood(s,u, Q_star, eta) * (kron_delta_xa - sa_likelihood(x,a, Q_star, eta) * kron_delta_x)
       
    #Eqn (8)
    T_pi = np.array([np.dot(pi_star[state], mdp.T[state]) for state in range(mdp.num_states)])         
    Tinv = np.linalg.inv(np.eye(mdp.num_states) - mdp.gamma * T_pi)
    #code for Rxa
    #temp = np.dot(mdp.T[s,u,:],Tinv[:,x])
    #delQsu_delRxa = kron_delta_xa + mdp.gamma *  temp * pi_star[x,a]
    #code for Rx trying it out...
    delQsu_delRxa = kron_delta_x + mdp.gamma *  np.dot(mdp.T[s,u,:], Tinv[:,x])
    
    #one element of Eqn (7)
    #print "delLsu_delQxa", delLsu_delQxa
    #print "delQsu_delRxa", delQsu_delRxa
    return delLsu_delQxa * delQsu_delRxa
    
def calc_reward_gradient(demo, mdp_r, R, eta=1.0):
    """calculate the gradient of the loglikelihood wrt reward"""
    num_states, num_actions = mdp_r.num_states, mdp_r.num_actions
    #solve mdp with given reward R
    mdp = copy.deepcopy(mdp_r)
    mdp.set_reward(R)
    pi_star, V_star = mdp_solver.policy_iteration(mdp)
    Q_star = mdp_solver.calc_qvals(mdp, pi_star, V_star, mdp.gamma)
    #calculate gradient
    gradient = np.zeros((num_states, num_actions))
    one_over_Lr = 1.0 / np.array([sa_likelihood(s,a,Q_star,eta) for s,a in demo])
    #print "one_over", one_over_Lr
    for x in range(num_states):
        for a in range(num_actions):
            delL_delRxa = np.array([partial_Lsu_partial_Rxa(s, u, x, a, mdp, 
                           Q_star, pi_star, eta) for s,u in demo])
            #print "delL_delR", delL_delRxa
            #print "debug", np.dot(one_over_Lr, delL_delRxa)
            gradient[x,a] = np.dot(one_over_Lr, delL_delRxa)
        
    return gradient

#eta = 1.0
#Q_star = np.array([[1,1,1,1],
#                   [1,1,1,1]])
#demo = [(0,0),(1,1)]
#print sa_likelihood(0,3,Q_star, eta)
#print demo_likelihood(demo, Q_star, eta)
#print demo_log_likelihood(demo, Q_star, eta)
