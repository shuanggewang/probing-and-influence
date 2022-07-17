import config
import copy

class State:
    def __init__(self, dynamics, lane):
        self.x = [0 for i in dynamics]
        self.v = [0 for i in dynamics]
        self.lane = lane
        for i in range(len(dynamics)):
            self.x[i], self.v[i] = dynamics[i]
            
    def update(self, controls, d_t):
        new_dynamics = [0 for i in controls]
        
        for i in range(len(controls)):
            new_dynamics[i] = self.x[i] + controls[i] * d_t**2 / 2 + self.v[i] * d_t, self.v[i] + d_t * controls[i]
            
        new_state = State(new_dynamics, self.lane)
        return new_state

    def print(self):
        print("positions", self.x)
        print("velocity", self.v)
        print("lane", self.lane)
        print("------------")
