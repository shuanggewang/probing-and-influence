from cmath import e
from hashlib import new
from this import s
import utils
import config
import copy
import numpy as np
from scipy.stats import entropy
from utils import Struct

class Ego:
    def __init__(self):
        self.id = 0
        self.phase = 0
        self.head_id = None
        return
        
    def find_headway(self, state):
        headway = []
        for i in range(2, len(state.x)):
            h = state.x[i] - state.x[0]
            if h >= 0:
                headway.append(h)
            else:
                break
        print(headway)
        return np.argmin(headway) + 2

    def generate_control(self, state):
        match self.phase:
            case 0:
                if self.head_id == None:
                    self.head_id = self.find_headway(state)
                print("huh",  self.head_id, state.x[self.head_id] - state.x[self.id])
                if ((state.x[self.head_id] - state.x[self.id]) >= 0):
                    state.lane[0] = 0
                    self.phase = 1
                return 0
            case 1:
                if(state.lane[1]==0):
                    self.head_id = 1
                    self.phase = 2
                reward = []
                for i in config.ego_action_space:
                    controls = [0 for i in range(len(state.x))]
                    controls[self.id] = i
                    new_state = state.update(controls, config.d_t)
                    reward.append(-(new_state.x[self.head_id] - new_state.x[self.id] - 60)**2 -(new_state.v[self.head_id] - new_state.v[self.id])**2)
                pos = np.argmax(reward)
                return config.ego_action_space[pos]
            case 2:
                reward = []
                for i in config.ego_action_space:
                    controls = [0 for i in range(len(state.x))]
                    controls[self.id] = i
                    new_state = state.update(controls, config.d_t)
                    reward.append(-(new_state.x[self.head_id] - new_state.x[self.id] - 30)**2 -(new_state.v[self.head_id] - new_state.v[self.id])**2)
                pos = np.argmax(reward)
                return config.ego_action_space[pos]
                


"""
    def generate_control(self, features, state, belief):
        OPT = {}
        for t in range(config.Horizon):
            print(t)
            struct_list = []
            for ego_action in config.ego_action_space:
                if t == 0:
                    belief_list = []
                    state_list = []
                    for i in range(len(belief.particles)):
                        #predict human control
                        human_action = self.predict_human_control(features, state, ego_action, belief.particles[i])
                        
                        #append belief
                        new_belief = belief.update(features, state, ego_action, human_action, config.d_t_predict)
                        belief_list.append(new_belief)
                        
                        #append state
                        next_state = state.update(ego_action, human_action, config.d_t_predict)
                        state_list.append(next_state)
                        
                    reward = np.dot([belief.entropy() for new_belief in belief_list], belief.prob) - np.dot([new_belief.entropy() for new_belief in belief_list], belief.prob)
                    struct = Struct(ego_action, state_list, belief_list, reward, None)
                else:
                    reward_sum = []
                    list_of_belief_list = []
                    list_of_state_list = []
                    for prev_struct in OPT[t-1]:
                        belief_list = []
                        state_list = []
                        for i in range(len(belief.particles)):
                            
                            #predict human control
                            human_action = self.predict_human_control(features, prev_struct.state[i], ego_action, belief.particles[i])
                            
                            #append belief
                            new_belief = prev_struct.belief[i].update(features, prev_struct.state[i], ego_action, human_action, config.d_t_predict)
                            belief_list.append(new_belief)
                            
                            #append state
                            next_state = prev_struct.state[i].update(ego_action, human_action, config.d_t_predict)
                            state_list.append(next_state)
        
                        reward = np.dot([old_belief.entropy() for old_belief in prev_struct.belief], belief.prob) - np.dot([new_belief.entropy() for new_belief in belief_list], belief.prob)
                        reward_sum.append(prev_struct.reward + reward)
                        list_of_state_list.append(state_list)
                        list_of_belief_list.append(belief_list)

                    pos = np.argmax(reward_sum)
                    struct = Struct(ego_action, list_of_state_list[pos], list_of_belief_list[pos], reward_sum[pos], OPT[t-1][np.argmax(reward_sum)])
                
                struct_list.append(struct)
            OPT[t] = struct_list
        # backtracking
        obj = max(OPT[config.Horizon - 1], key=lambda item: item.reward)
        list = []
        while(obj != None):
            print(obj.reward)
            list.insert(0, obj.control)
            obj = obj.prev
        return list
"""
                
                                     