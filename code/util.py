#utility functions

def print_grid(mdp, vals):
    for r in (range(mdp.rows)):
        for c in (range(mdp.cols)):
            print("%.2f " % vals[r][c]),
        print(" ")

       
def print_reward(mdp):
    print_grid(mdp, mdp.reward)



        
def print_policy(mdp, pi):
    count = 0
    for r in (range(mdp.rows)):
        #print r
        for c in (range(mdp.cols)):
            #print c
            if count in mdp.terminals:
                print '.',
            else:
                if pi[count,0] > 0:    #down
                    print 'v',
                elif pi[count,1] > 0:  #up
                    print '^',
                elif pi[count,2] > 0:  #left
                    print '<',
                else:                   #right
                    print '>',
            count += 1
        print(" ")
