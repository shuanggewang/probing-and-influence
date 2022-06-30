import math
import config
import numpy as np

class IDM:
    def __init__(self):
        self.v_0 = 20
        self.T = 1.5
        self.a = 0.73
        self.b = 1.67
        self.delta = 4
        self.s_0 = 2

    def generate_control(self, state):
        temp = self.s_0+state.v_h* self.T+(state.v_h - state.v_r) / (2 * math.sqrt(self.a*self.b))
        acc = self.a*(1-(state.v_h/self.v_0)**self.delta - (temp/state.headway)**2)
        return acc


