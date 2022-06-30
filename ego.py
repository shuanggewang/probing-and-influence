from cmath import e
from hashlib import new
import utils
import config
import copy
import numpy as np
from scipy.stats import entropy
from utils import Struct

class Ego:
    def __init__(self):
        return
    
    def predict_human_control(self, features, state, u_r, phi):
        reward_list=[]
        for u_h in config.human_action_space:
            next_state = state.update(u_r, u_h, config.d_t_predict)
            reward_list.append(features.weighted_sum(next_state, phi))
        pos = np.argmax(reward_list)
        return reward_list[pos]
    
    """
    def generate_control(self, features, state, belief):
        OPT = {}
        for t in range(config.Horizon):
            print(t)
            struct_list = []
            for ego_action in config.ego_action_space:
                if t == 0:
                    human_action_list=[]
                    for i in range(len(belief.particles)):
                        
                        #predict human control
                        temp_human_action = self.predict_human_control(features, state, ego_action, belief.particles[i])
                        human_action_list.append(temp_human_action)

                    human_action = np.dot(human_action_list, belief.prob)
                    
                    next_state = state.update(ego_action, human_action, config.d_t_predict)
                        
                    new_belief = belief.update(features, state, ego_action, human_action, config.d_t_predict)
                    
                    reward = belief.entropy() - new_belief.entropy()
                    print(ego_action, reward)
                    struct = Struct(ego_action, next_state, new_belief, reward, None)
                else:
                    reward_sum = []
                    for prev_struct in OPT[t-1]:
                        human_action_list=[]
                        for i in range(len(belief.particles)):
                            
                            #predict human control
                            temp_human_action = self.predict_human_control(features, prev_struct.state, ego_action, belief.particles[i])
                            human_action_list.append(temp_human_action)

                        human_action = np.dot(human_action_list, belief.prob)
                        
                        #append state
                        next_state = prev_struct.state.update(ego_action, human_action, config.d_t_predict)
                        
                        #append belief
                        new_belief = prev_struct.belief.update(features, prev_struct.state, ego_action, human_action, config.d_t_predict)
    
                        reward = prev_struct.belief.entropy() - new_belief.entropy()
                        reward_sum.append(prev_struct.reward + reward)
                        
                    pos = np.argmax(reward_sum)
                    new_action = config.ego_action_space[pos]
                    new_state = OPT[t-1][pos].state.update(new_action, human_action, config.d_t_predict)
                    new_belief = OPT[t-1][pos].belief.update(features, OPT[t-1][pos].state, ego_action, human_action, config.d_t_predict)
                    struct = Struct(new_action, new_state, new_belief, reward_sum[pos], OPT[t-1][np.argmax(reward_sum)])
                
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
    def generate_control(self, features, state, belief):
        new_belief = copy.deepcopy(belief)
        ego_action_list = []
        human_action_list = []
        for i in config.ego_action_space:
            ego_action_list.append([i for index in range(config.Horizon)])
        reward_list=[]

        for ego_action in ego_action_list:
            print(ego_action)
            entropy_list=[]
            for i in range(len(new_belief.particles)):
                human_action = utils.dynamic_programming(features, state, ego_action, new_belief.particles[i])
                for j in range(len(ego_action)):
                    new_belief.update(features, state, ego_action[j], human_action[j])
                entropy_list.append(new_belief.entropy())
            reward_list.append(np.dot(new_belief.prob, entropy_list))
            print(reward_list)
        return ego_action[np.argmin(reward_list)]
"""
   
                
                                     