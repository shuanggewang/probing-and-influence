import math
import config
import numpy as np

class Idm:
    def __init__(self):
        self.v_0 = 20
        self.T = 1.5
        self.a = 0.73
        self.b = 1.67
        self.delta = 4
        self.s_0 = 2
        self.id = 1
        self.head_id = None
        self.phase = 0
    
    def find_headway(self, state):
        headway = []
        for i in range(2, len(state.x)):
            h = state.x[i]-state.x[1]
            if h <= 0 :
                break
            else:
                headway.append(h)
        return np.argmin(headway) + 2
            
    def generate_control(self, state):
        if self.head_id == None:
            self.head_id = self.find_headway(state)

        min_headway = self.s_0+state.v[1]* self.T+(state.v[1] - state.v[self.head_id]) / (2 * math.sqrt(self.a*self.b))
        real_headway = state.x[self.head_id] - state.x[self.id]
        
        print(min_headway, real_headway)
        
        if real_headway <= min_headway:
            reward = []
            for i in config.human_action_space:
                controls = [0 for i in range(len(state.x))]
                controls[self.id] = i
                new_state = state.update(controls, config.d_t)
                reward.append(-2*(new_state.x[self.id] - new_state.x[0] - 15)**2 + -(new_state.v[self.id] - new_state.v[0])**2)
            pos = np.argmax(reward)
            return config.human_action_space[pos]
        else:
            print("reached")
            state.lane[self.id] = 0
            desired_headway = self.s_0+state.v[1]* self.T+(state.v[1] - state.v[self.head_id]) / (2 * math.sqrt(self.a*self.b))
            human_action = self.a*(1-(state.v[1]/self.v_0)**self.delta - (desired_headway/(np.Inf))**2)
            return human_action
        


