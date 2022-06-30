import config

class State:
    def __init__(self, x_r, x_h, v_r, v_h, u_r, u_h, jerk):
        self.x_r = x_r
        self.x_h = x_h
        self.v_r = v_r
        self.v_h = v_h
        self.u_r = u_r
        self.u_h = u_h
        self.jerk = jerk
        self.headway = x_r - x_h
        self.v_diff = v_r - v_h

    def update(self, u_r, u_h, d_t):
        new_x_r = self.x_r + u_r * d_t**2 / 2 + self.v_r * d_t
        new_x_h = self.x_h + u_h * d_t**2 / 2 + self.v_h * d_t
        new_v_r = self.v_r + d_t * u_r
        new_v_h = self.v_h + d_t * u_h
        new_jerk = (u_h - self.u_h)/d_t
        new_state = State(new_x_r, new_x_h, new_v_r, new_v_h, u_r, u_h, new_jerk)
        return new_state
    
    def print(self):
        print("headway:", self.headway, "v_r:", self.v_r, "v_h:", self.v_h,"u_r:", self.u_r, "u_h:", self.u_h )
