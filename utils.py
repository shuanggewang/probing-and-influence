import config
import numpy as np


class Struct:
    def __init__(self, control, state, belief, reward, prev):
        self.control = control
        self.state = state
        self.belief = belief
        self.reward = reward
        self.prev = prev

def state_to_plot(state):
    relative = []
    for i in range(len(state.x)):
        relative.append((state.x[i] - state.x[0])*-config.plot_to_real_ratio)
    relative = [each + 600 for each in relative]
    return relative
    
    


"""
def dynamic_programming(features, state, u_r, phi):
    OPT = {}
    for t in range(config.Horizon):
        struct_list = []
        for u_h in config.action_space:
            if t == 0:
                next_state = state.update(u_r[t], u_h, config.d_t_predict)
                struct = Struct(next_state, features.weighted_sum(next_state, phi), None)
            else:
                reward_sum = []
                for prev_struct in OPT[t-1]:
                    next_state = prev_struct.state.update(u_r[t], u_h, config.d_t_predict)
                    reward_sum.append(prev_struct.reward + features.weighted_sum(next_state, phi))
                struct = Struct(OPT[t-1][np.argmax(reward_sum)].state.update(0, u_h, config.d_t_predict), reward_sum[np.argmax(reward_sum)], OPT[t-1][np.argmax(reward_sum)])
            struct_list.append(struct)
        OPT[t] = struct_list
    # backtracking
    obj = max(OPT[config.Horizon - 1], key=lambda item: item.reward)
    list = []
    while(obj != None):
        list.insert(0, obj.state.u_h)
        obj = obj.prev
    return list
"""


def generate_phi(i, j):
    temp = [[(1/config.cols/config.rows)for i in range(config.cols)]
            for j in range(config.rows)]
    for x in range(config.rows):
        for y in range(config.cols):
            temp[x][y] = -((x-i)**2 + (y-j)**2)
    flat = np.array(temp).flatten()
    mini = min(flat)
    for x in range(config.rows):
        for y in range(config.cols):
            temp[x][y] -= mini
    flat = np.array(temp).flatten()
    summation = sum(flat)
    for x in range(config.rows):
        for y in range(config.cols):
            temp[x][y] /= summation
            temp[x][y] *= config.feature_norm
    return temp


def generate_particles():
    #temp = [(i, 6) for i in range(config.cols)]
    temp = [(i, 6) for i in range(config.cols)]
    particles = []
    for index in temp:
        phi = generate_phi(index[0], index[1])
        particles.append(phi)
    return particles
