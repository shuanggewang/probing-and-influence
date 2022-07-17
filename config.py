import numpy as np

d_t, d_t_predict = 0.1, 1

human_action_space = np.linspace(start=-1, stop=1, num=201)
ego_action_space = np.linspace(start=-1, stop=1, num=201)
acc_action_space = np.linspace(start=-1, stop=1, num=201)

Horizon = 5
sigma = 5
graph_gap = 5
feature_norm = 10
cols, rows = 20, 20

v_space = np.linspace(start=0, stop=36, num=cols)
h_space = np.linspace(start=0, stop=350, num=rows)


car_width, car_length = 3, 6

plot_to_real_ratio = 5

left_lane_center, right_lane_center = 50, 100
