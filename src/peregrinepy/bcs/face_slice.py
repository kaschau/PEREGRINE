from numpy import s_
fs = {}
#face1
fs['1'] = {'s0_' : s_[ 0,:,:],
           's1_' : s_[ 1,:,:],
           's2_' : s_[ 2,:,:]}
#face2
fs['2'] = {'s0_' : s_[-1,:,:],
           's1_' : s_[-2,:,:],
           's2_' : s_[-3,:,:]}
#face3
fs['3'] = {'s0_' : s_[:, 0,:],
           's1_' : s_[:, 1,:],
           's2_' : s_[:, 2,:]}
#face4
fs['4'] = {'s0_' : s_[:,-1,:],
           's1_' : s_[:,-2,:],
           's2_' : s_[:,-3,:]}
#face5
fs['5'] = {'s0_' : s_[:,:, 0],
           's1_' : s_[:,:, 1],
           's2_' : s_[:,:, 2]}
#face6
fs['6'] = {'s0_' : s_[:,:,-1],
           's1_' : s_[:,:,-2],
           's2_' : s_[:,:,-3]}
