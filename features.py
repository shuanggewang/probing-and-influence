import config
import numpy as np

class Features:
    def __init__(self):
        return
    
    def weighted_sum(self, state, phi):
        reward=[]
        for v in config.v_space:
            for h in config.h_space:
                r = np.exp(-((state.v_h - v )**2+((state.headway - h)**2))/config.sigma)
                reward.append(r)
        reward = np.array(reward).flatten()
        phi = np.array(phi).flatten()
        return np.dot(reward, phi)