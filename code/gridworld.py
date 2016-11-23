import numpy as np

class GridWorld():

    def __init__(self, reward, terminals, gamma):
        self.num_actions = 4
        #print self.num_actions
        self.rows = len(reward)
        #print self.rows
        self.cols = len(reward[0])
        #print self.cols
        self.num_states = self.cols * self.rows
        self.terminals = terminals
        self.T = self.build_trans_matrix(self.rows,self.cols,self.terminals)
        #print self.T
        self.reward = reward
        self.R = np.reshape(reward,(self.num_states,1))
        #print self.R
        self.gamma = gamma

    def build_trans_matrix(self, rows, cols, terminals):
        # rows-by-cols gridworld  
        #ex: if rows = cols = 5 then laid out like:
        # 0  1  2  3  4
        # 5  6  7  8  9 
        # ...
        # 20 21 22 23 24

        trans_mat = np.zeros((rows*cols, self.num_actions, rows*cols))

        # Action 0 = down
        for s in range(rows * cols):
            if s < (rows - 1) * cols:
                trans_mat[s,0,s+cols] = 1
            else:
                trans_mat[s,0,s] = 1
              
        # Action 1 = up
        for s in range(rows * cols):
            if s >= cols:
                trans_mat[s,1,s-cols] = 1
            else:
                trans_mat[s,1,s] = 1
              
        # Action 2 = left
        for s in range(rows * cols):
            if s%cols > 0:
                trans_mat[s,2,s-1] = 1
            else:
                trans_mat[s,2,s] = 1
              
        # Action 3 = right
        for s in range(rows * cols):
            if s%cols < (cols - 1):
                trans_mat[s,3,s+1] = 1
            else:
                trans_mat[s,3,s] = 1
              
        for s in terminals:
            trans_mat[s,:,:] = np.zeros((self.num_actions, rows * cols))
          
        return trans_mat
        
    def set_reward(self, R):
        """input: a |S|x1 vector of rewards"""
        self.R = R
        self.reward = np.reshape(R, (self.rows,self.cols))

