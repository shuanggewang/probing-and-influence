import numpy as np

d_t, d_t_predict = 0.1, 1

human_action_space = np.linspace(start = -1, stop = 1, num = 3)
ego_action_space = np.linspace(start = -1, stop = 1, num = 3)

Horizon = 5
sigma = 5
graph_gap = 5
feature_norm = 1
cols, rows = 30, 30
v_space = np.linspace(start=0, stop=36, num=cols)
h_space = np.linspace(start=0, stop=350, num=rows)
