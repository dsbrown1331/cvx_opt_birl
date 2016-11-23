import numpy as np

#mdp module for solving mdps


def policy_iteration(mdp):
    num_states = mdp.num_states
    num_actions = mdp.num_actions
    #start with arbitrary policy of one action (down) for all states
    pi = np.hstack((np.ones((num_states,1)),np.zeros((num_states, num_actions-1))))
    count = 0
    while True:
        #print "---iter", count, "-----"
        count += 1
        #solve linear equations
        V_pi = calc_policy_value(mdp, pi)
        #print "policy value", V_pi
        
        #improve policy at each state
        new_pi = improve_policy(mdp, V_pi)
        
        if np.allclose(pi,new_pi):
            break
        else:
            pi = new_pi
    
    return pi, calc_policy_value(mdp,pi)  

def calc_policy_value(mdp, pi):
    T_pi = np.array([np.dot(pi[s], mdp.T[s]) for s in range(mdp.num_states)])
    #print "policy transitions", T_pi         
    V_pi = np.linalg.solve((np.eye(mdp.num_states) - mdp.gamma * T_pi), mdp.R)
    return V_pi
    
def improve_policy(mdp, V_pi):
        values = np.hstack([mdp.R + mdp.gamma * np.dot(mdp.T[:,a,:], V_pi) for a in range(mdp.num_actions)])
        #print "values", values
        
        pi_actions = np.argmax(values,axis=1)
        #print "argmax actions", pi_actions
        new_pi = np.zeros((mdp.num_states, mdp.num_actions))
        for s in range(mdp.num_states):
            new_pi[s,pi_actions[s]] = 1.0
        #print "new policy", new_pi
        return new_pi
        
 
def calc_qvals(mdp, pi, V_pi, gamma):
    """
    input T[s,a,s'] = transition probability for (s,a,s') tuple
    R[s] = reward in state s
    Pi[s,a] = policy probability of taking action in state s
    gamma = discount factor
    """
    T, R = mdp.T, mdp.R
    num_states = T.shape[0]
    num_actions = T.shape[1]
    R_sa = np.outer(R,np.ones(num_actions))
    #T_Pi[s,s'] probability of ending up in state s' from s when following policy Pi
    Q_pi = R_sa + gamma * np.dot(T,V_pi)[:,:,0]
    return Q_pi



