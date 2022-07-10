import math
import config
import numpy as np


class Acc:
    def __init__(self, id):
        self.id = id
        self.headway = 15
        self.v_0 = 20
        self.head_id = id - 1


    def generate_control(self, state):
        if self.id == 2:
            return 0
        else:
            print(self.head_id, self.id)
            headway = state.x[self.head_id] - state.x[self.id]
        if (headway <= self.headway):
            reward = []
            for i in config.acc_action_space:
                controls = [0 for i in range(len(state.x))]
                controls[self.id] = i
                new_state = state.update(controls, config.d_t)
                reward.append(-(new_state.x[self.head_id] - new_state.x[self.id] - self.headway)**2)
            pos = np.argmax(reward)
            return config.acc_action_space[pos]
        else:
            reward = []
            for i in config.acc_action_space:
                controls = [0 for i in range(len(state.x))]
                controls[self.id] = i
                new_state = state.update(controls, config.d_t)
                reward.append(-(new_state.v[self.head_id] - new_state.v[self.id])**2 - 10*(new_state.x[self.head_id] - new_state.x[self.id] - self.headway)**2)
            pos = np.argmax(reward) 
            return config.acc_action_space[pos]