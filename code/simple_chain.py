import numpy as np

class SimpleChain():
    #simple chain with deterministic actions left and right
    def __init__(self, reward, terminals, gamma):
        self.num_actions = 2
        #print self.num_actions
        self.num_states = len(reward)
        #print self.rows
        self.terminals = terminals
        self.T = self.build_trans_matrix(self.num_states,self.terminals)
        #print self.T
        self.reward = reward
        self.R = np.reshape(reward,(self.num_states,1))
        #print self.R
        self.gamma = gamma

    def build_trans_matrix(self, num_states, terminals):
        # simple markov chain
        # 0  1  2  3  4 ....
        #with transitions left and right

        trans_mat = np.zeros((num_states, self.num_actions, num_states))

        # Action 0 = left
        for s in range(num_states):
            if s == 0:
                trans_mat[s,0,s] = 1
            else:
                trans_mat[s,0,s-1] = 1
              
        # Action 1 = right
        for s in range(num_states):
            if s == (num_states-1):
                trans_mat[s,1,s] = 1
            else:
                trans_mat[s,1,s+1] = 1
              
        #can't escape      
        for s in terminals:
            trans_mat[s,:,:] = np.zeros((self.num_actions, num_states))
          
        return trans_mat
        
    def set_reward(self, R):
        """input: a |S|x1 vector of rewards"""
        self.R = R
        self.reward = np.reshape(R, (1,self.num_states))


if __name__=="__main__":
    #testing stuff
    reward = [0, 0, 1]
    terminals = []
    gamma = 0.9
    chain = SimpleChain(reward, terminals, gamma)
    print "reward"
    print chain.reward
    print chain.R
    print "transitions"
    print chain.T
