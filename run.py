
import config
import utils
from state import State
from idm import Idm
from features import Features
from ego import Ego
from acc import Acc
from visual import *

import numpy as np
import matplotlib.pyplot as plt


particles = utils.generate_particles()
features = Features()
ego = Ego()
idm = Idm()
cars = 8
acc_list = []

for i in range(2, cars):
    acc_list.append(Acc(i))

dynamics = [[0, 20], [-15, 20], [45, 20], [30, 20],
            [15, 20], [0, 20], [-15, 20], [-30, 20]]

lanes = [1, 1] + [0 for i in range(2, cars-2)]

state = State(dynamics, lanes)

# lanes
l1, l2 = Lane([25, 0], 50, 1), Lane([75, 0], 50, 2)

lane_list = [l1, l2]

pos = utils.state_to_plot(state)

# agents
ego_P, ego_S = [config.right_lane_center, pos[0]], 19
human_P, human_S = [config.right_lane_center, pos[1]], 19


robot = Agent('./images/car-yellow.png', ego_P, ego_S)
human = Agent('./images/car-orange.png', human_P, human_S)

agent_list = [robot, human]

# obstacles
obj_list = []

for i in range(2, cars):
    obj_list.append(Obstacles('./images/car-white.png',
                    [config.left_lane_center, pos[5]], 20))


if __name__ == '__main__':

    # plot to real world: 10 : 1
    env = Environment(lane_list, obj_list, agent_list, state)
    gameExit = False

    timer = 0

    x, y1, y2, y3 = [], [], [], []

    plt.ion()
    figure = plt.figure()

    while not gameExit:

        controls = []

        print("huh", ego.head_id)
        if(state.lane[0] == 0):
            acc_list[ego.head_id-1].head_id = 0

        # ego control
        ego_control = ego.generate_control(state)
        controls.append(ego_control)

        # human control
        human_action = idm.generate_control(state)
        controls.append(human_action)

        # acc control
        for acc in acc_list:
            controls.append(acc.generate_control(state))

        state.print()

        print("controls", controls)

        new_state = state.update(controls, config.d_t)

        env.step(new_state, state)

        state = new_state

        if timer == 0:
            x.append(timer/10)
            y1.append(controls[1]*config.d_t)
            y2.append(controls[6]*config.d_t)
            y3.append(controls[7]*config.d_t)
        else:
            x.append(timer/10)
            y1.append(y1[-1]+controls[1]*config.d_t)
            y2.append(y2[-1]+controls[6]*config.d_t)
            y3.append(y3[-1]+controls[7]*config.d_t)


        ax = figure.add_subplot(111)

        ax.step(x, y1, where='mid', label='Human Vehicle (Orange)')
        
        ax.step(x, y2, where='mid', label='Rear Vehicle 1 (White)')

        ax.step(x, y3, where='mid', label='Rear Vehicle 2 (White)')
        
        ax.grid(axis='x', color='0.95')
        ax.legend(title='Vehicles')
        ax.set_xlabel('$Time \hspace{1} (s)$')
        ax.set_ylabel('$Cumulative \hspace{1} Deviation \hspace{1} (m/s)$')
        figure.canvas.draw()
        
        if (min(state.v)>=19 and timer >= 400):
            figure.savefig("./figures/deviation1.jpg", dpi=300)
            input()
        
        figure.canvas.flush_events()
        figure.clf()

        print("current time:", timer/10)

        timer += 1
        

            
        mainClock.tick(FPS)
        # input()

    pygame.quit()
    quit()
