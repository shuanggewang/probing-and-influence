
import config
import utils
from state import State
from idm import Idm
from features import Features
from ego import Ego
from acc import Acc
from visual import *


particles = utils.generate_particles()
features = Features()
ego = Ego()
idm = Idm()
cars = 8
acc_list = []
for i in range(2, cars):
    acc_list.append(Acc(i))

dynamics = [[0, 20], [-15, 20], [45, 20], [30, 20], [15, 20], [0, 20], [-15, 20], [-30, 20]]
lanes = [1,1] + [0 for i in range(2,cars-2)]
state = State(dynamics, lanes)

# lanes
l1, l2 = Lane([150, 0], 100, 1), Lane([250, 0], 100, 2)

lane_list = [l1, l2]

pos = utils.state_to_plot(state)

# agents
ego_P, ego_S = [300, pos[0]], 19
human_P, human_S = [300, pos[1]], 19


robot = Agent('./images/car-yellow.png', ego_P, ego_S)
human = Agent('./images/car-orange.png', human_P, human_S)

agent_list = [robot, human]

# obstacles
obj_list = []

for i in range(2, cars):
    obj_list.append(Obstacles('./images/car-white.png', [200, pos[5]], 20))


if __name__ == '__main__':

    # plot to real world: 10 : 1
    env = Environment(lane_list, obj_list, agent_list, state)
    gameExit = False
    
    timer = 0
    while not gameExit:

        controls = []
        
        print("huh",ego.head_id)
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
        
        timer += 1
        
        print("current time:", timer/10)
        

        mainClock.tick(FPS)
        # input()

    pygame.quit()
    quit()
