from pickletools import optimize
import time
import math
import numpy as np
import matplotlib.pyplot as plt

import config
import utils
from state import State
from belief import Belief
from idm import IDM
from features import Features
from ego import Ego


particles = utils.generate_particles()
idm = IDM()
belief = Belief(particles, [1/len(particles) for i in range(len(particles))])
features = Features()
ego = Ego()
state = State(100, 0, 15, 20, 0, 0, 0)
x = [r'$\varphi_{'+str(i)+'}$' for i in range(config.rows)]

plt.ion()
figure = plt.figure()

counter = 0
image = 0

stacking = []

if __name__ == "__main__":
    while(True):
        u_h = [idm.generate_control(state)]
        if counter < 50:
            u_r = [0]
        elif counter < 100:
            if (counter % 10 ==0):
                u_r = ego.generate_control(features, state, belief)
        else:
            counter = 0
            
        print(u_r, counter, image)
        state = state.update(u_r[0], u_h[0], config.d_t)
        ax = figure.add_subplot(111)
        ax.set_xticks([i*config.graph_gap for i in range(int(config.rows/config.graph_gap))])
        ax.set_xticklabels([r'$\varphi_{'+str(i*config.graph_gap)+'}$' for i in range(int(config.rows/config.graph_gap))])
        ax.bar(x, belief.prob)
        ax.set_ylabel('$Probability$')
        figure.canvas.draw()
        if counter == 0:
            stacking.append(belief.prob)
            if(len(stacking)==1):
                pass
            else:
                utils.plot_stacking(stacking[1:])
            figure.savefig("./figures/probing_{}.jpg".format(image), dpi=300)
            image +=1
        print(belief.prob)
        figure.canvas.flush_events()
        figure.clf()
        belief = belief.update(features, state, u_r[0], u_h[0], config.d_t)
        state.print()
        counter += 1