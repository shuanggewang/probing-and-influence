import math
import config
import numpy as np


class Idm:
    def __init__(self):
        self.v_0 = 25
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
            h = state.x[i] - state.x[1]
            if h <= 0:
                break
            else:
                headway.append(h)
        return np.argmin(headway) + 2
    
    def merge_condition(self, state):
        head_id = self.find_headway(state)
        condition1 = state.v[1] < 17
        condition2 = (state.x[head_id] - state.x[self.id]) > 10
        condition3 = (state.x[self.id]- state.x[head_id+1]) > 40
        return condition1 and condition2 and condition3

    def generate_control(self, state):
        match self.phase:
            case 0:
                if self.head_id == None:
                    self.head_id = 0
                if (self.merge_condition(state)):
                    self.phase = 1
                    state.lane[1] = 0
                    self.head_id = self.find_headway(state)
                desired_headway = self.s_0 + state.v[1] * self.T + (state.v[1] - state.v[self.head_id]) / (2 * math.sqrt(self.a * self.b))
                human_action = self.a * (1 - (state.v[1] / self.v_0)**self.delta - (desired_headway / (state.x[self.head_id] - state.x[self.id]))**2)
                return human_action
            case 1:
                desired_headway = self.s_0 + state.v[1] * self.T + (state.v[1] - state.v[self.head_id]) / (2 * math.sqrt(self.a * self.b))
                human_action = self.a * (1 - (state.v[1] / self.v_0)**self.delta - (desired_headway / (state.x[self.head_id] - state.x[self.id]))**2)
                return human_action
                
                
