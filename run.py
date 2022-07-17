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

cars = 7
acc_list = []

headway = [50, 50, 50, 80, 50]
for i in range(2, cars):
    acc_list.append(Acc(i, headway[i - 2]))

dynamics = [[0, 19], [-50, 19],
            [0, 20], [-50, 20], [-100, 20], [-180, 20], [-230, 20]]

lanes = [1, 1] + [0 for i in range(2, cars - 2)]

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

    x, y1, y2 = [], [], []

    plt.ion()
    figure = plt.figure()

    while not gameExit:

        controls = []

        #print("target:" , ego.target, "phase:", ego.phase)
        #print(acc_list[5].head_id, "sdsd")
        if(state.lane[1] == 0):
            acc_list[idm.head_id -1].head_id = 1

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

        """
        if timer == 0:
            x.append(timer/10)
            y1.append(abs(controls[6])*config.d_t)
            y2.append(abs(controls[7])*config.d_t)
        else:
            x.append(timer/10)
            y1.append(y1[-1]+abs(controls[6])*config.d_t)
            y2.append(y2[-1]+abs(controls[7])*config.d_t)

        ax = figure.add_subplot(111)

        #ax.set_xticks([i*config.graph_gap for i in range(int(config.rows/config.graph_gap))])
        #ax.set_xticklabels([r'$\varphi_{'+str(i*config.graph_gap)+'}$' for i in range(int(config.rows/config.graph_gap))])

        ax.step(x, y1, where='mid', label='Rear Vehicle 1')

        ax.step(x, y2, where='mid', label='Rear Vehicle 2')

        ax.grid(axis='x', color='0.95')
        ax.legend(title='Rear Vehicles #')
        ax.set_xlabel('$Time \hspace{1} (s)$')
        ax.set_ylabel('$Cumulative \hspace{1} Deviation \hspace{1} (m/s)$')
        figure.canvas.draw()
        
        if (min(state.v)>=19 and timer >= 400):
            figure.savefig("./figures/deviation.jpg", dpi=300)
            input()
        
        figure.canvas.flush_events()
        figure.clf()
        """
        
        if (min(state.v)>=19 and timer >= 400):
            figure.savefig("./figures/deviation.jpg", dpi=300)
            input()
        
        
        print("current time:", timer/10)

        timer += 1
        

            
        mainClock.tick(FPS)
        # input()

    pygame.quit()
    quit()
