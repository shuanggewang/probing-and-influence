import config
import numpy as np
from scipy.stats import entropy

class Belief:
    def __init__(self, particles, prob):
        self.particles = particles
        self.prob = prob

    def update(self, features, state, u_r, u_h, d_t):
        next_state = state.update(u_r, u_h, d_t)
        p = [0 for i in range(len(self.particles))]
        for i in range(len(self.particles)):
            p[i] = self.prob[i] * np.exp(features.weighted_sum(next_state, self.particles[i]))
        summation = sum(p)
        for i in range(len(self.particles)):
            p[i] /= summation
        new_belief = Belief(self.particles, p)
        return new_belief

    def entropy(self):
        return entropy(self.prob)
