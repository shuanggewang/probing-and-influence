import config
import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class Struct:
    def __init__(self, control, state, belief, reward, prev):
        self.control = control
        self.state = state
        self.belief = belief
        self.reward = reward
        self.prev = prev


"""
class Struct:
    def __init__(self, state, reward, prev):
        self.state = state
        self.reward = reward
        self.prev = prev
"""
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
    temp = [[(1/config.cols/config.rows)for i in range(config.cols)]for j in range(config.rows)]
    for x in range(config.rows):
        for y in range(config.cols):
            #temp[x][y] = -((x-i)**2 + (y-j)**2)
            temp[x][y] = np.exp(-((x-i)**2 + (y-j)**2)/10)
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
    temp = [(i, 9) for i in range(config.cols)]
    particles = []
    for index in temp:
        phi = generate_phi(index[0], index[1])
        particles.append(phi)
    return particles

def plot_stacking(stacking):
    x = [r'$\varphi_{'+str(i)+'}$' for i in range(config.rows)]
    stacked = plt.figure()
    kwargs = dict(alpha=0.3,  ec="k")
    ax = stacked.add_subplot(111)
    for i in range(len(stacking)):
        ax.bar(x, stacking[i], **kwargs, label="{} s".format((i+1)*15), linewidth = 0, width=1)
    ax.legend(loc='upper right')
    ax.set_xticks([i*config.graph_gap for i in range(int(config.rows/config.graph_gap))])
    ax.set_xticklabels([r'$\varphi_{'+str(i*config.graph_gap)+'}$' for i in range(int(config.rows/config.graph_gap))])
    ax.set_ylabel('$Probability$')
    stacked.savefig("./figures/probing_velocity_{}.jpg".format(len(stacking)), dpi=300)

if __name__ == "__main__":
    particles = generate_particles()
    
    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')
    x = y = np.linspace(0, config.cols, config.cols)
    X, Y = np.meshgrid(x, y)
    Z = np.array(particles[15])
    my_col = cm.jet(Z)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0.001, vmin=0, vmax=0.03)
    #ax.set_zlim(0, 0.03)
    figure.colorbar(surf, location= "left", shrink=0.5, aspect=5)
    #figure.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('Headway')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Probability')
    plt.savefig("./figures/2d.jpg", dpi=300)
    plt.show()
