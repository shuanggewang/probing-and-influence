import config
import numpy as np


class Struct:
    def __init__(self, control, state, belief, reward, prev):
        self.control = control
        self.state = state
        self.belief = belief
        self.reward = reward
        self.prev = prev

def state_to_plot(state):
    relative = []
    for i in range(len(state.x)):
        relative.append((state.x[i] - state.x[0])* - config.plot_to_real_ratio)
    relative = [each + 50 for each in relative]
    return relative

def generate_phi(i, j):
    temp = [[(1/config.cols/config.rows)for i in range(config.cols)]
            for j in range(config.rows)]
    for x in range(config.rows):
        for y in range(config.cols):
            temp[x][y] = -((x-i)**2 + (y-j)**2)
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
    temp = [(i, 6) for i in range(config.cols)]
    particles = []
    for index in temp:
        phi = generate_phi(index[0], index[1])
        particles.append(phi)
    return particles


def images_to_video():
    import cv2
    import numpy as np
    import os
    from os.path import isfile, join
    pathIn= './figures/'
    pathOut = './videos/video.avi'
    fps = 40
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[x.index("_")+1:x.index(".")]))
    for i in range(len(files)):
        filename = pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        height, width, layers = img.shape
        size = (width,height)
        
        #inserting the frames into an image array
        frame_array.append(img)
    
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


if __name__ == '__main__':
    images_to_video()