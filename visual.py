
from typing import Counter
import pygame
import matplotlib.path as mplPath
import numpy as np
import math
import time
import copy
from scipy.spatial.distance import euclidean as point_dist
import random

import utils
import config


WIDTH = 150
HEIGHT = 1000
FPS = 40
count = 0


pygame.init()
displaySurface = pygame.display.set_mode((WIDTH, HEIGHT))
mainClock = pygame.time.Clock()
pygame.display.set_caption('Traffic World')
pygame.display.update()

# ------------------------------ Classes ----------------------------------------------------------


class Lane(object):
    def __init__(self, corner, width, number):
        self.corner = corner
        self.width = width
        self.middle = corner[0] + self.width/2.0
        self.vertices = self._calculateVertices()
        self.color = [128, 128, 128]
        self.number = number

        self.bound_l = [self.corner, [self.corner[0], HEIGHT]]
        self.bound_r = [[self.corner[0]+self.width,
                         self.corner[1]], [self.corner[0]+self.width, HEIGHT]]
        self.bound_color = [0, 0, 0]

        self.lane_rect = pygame.draw.polygon(
            displaySurface, self.color, self.vertices)

        self._markerWidth = 5
        self._markerLen = 50
        self._markerGap = 30
        self._markerNum = HEIGHT / (self._markerLen + self._markerGap)
        self.markerList = list()

        self.bushList = list()

        for i in range(int(self._markerNum) + 1):
            v1 = [self.bound_r[0][0] - self._markerWidth/2,
                  self.bound_r[0][1] + i * (self._markerLen + self._markerGap)]
            v2 = [self.bound_r[0][0] + self._markerWidth/2,
                  self.bound_r[0][1] + i * (self._markerLen + self._markerGap)]
            v3 = [self.bound_r[0][0] + self._markerWidth /
                  2, v2[1] + self._markerLen]
            v4 = [self.bound_r[0][0] - self._markerWidth /
                  2, v2[1] + self._markerLen]

            self.marker_vertices = [v1, v2, v3, v4]
            marker = pygame.draw.polygon(
                displaySurface, [255, 255, 255], self.marker_vertices)
            self.markerList.append(marker)

            bush_img = pygame.image.load('./images/tree.png')
            bush_img = pygame.transform.scale(bush_img, (30, 30))
            bush_rect = bush_img.get_rect()

            if self.number == 1:
                bush_rect.center = [75, v1[1]]
            if self.number == 2:
                bush_rect.center = [WIDTH-75, v1[1]]

            self.bushList.append([bush_img, bush_rect])

    def _calculateVertices(self):
        v1 = self.corner
        v2 = [self.corner[0]+self.width, self.corner[1]]
        v3 = [v2[0], HEIGHT]
        v4 = [self.corner[0], HEIGHT]
        return [v1, v2, v3, v4]

    def relative_move(self, vel):

        new_obj = list()

        for m in self.markerList:
            o = m.move(vel[0], vel[1])

            if o.top > HEIGHT:
                o.top = o.top % HEIGHT

            elif o.top < 0:
                o.top = HEIGHT

            new_obj.append(o)

        self.markerList = new_obj


class Obstacles(object):
    def __init__(self, image_name, position, initial_speed):
        self.image_name = image_name
        self.img = pygame.image.load(image_name)
        self.img = pygame.transform.scale(self.img, (config.car_width* 10, config.car_length* 10))
        self.mutable_img = self.img.copy()

        self.boundingBox = self.img.get_rect()
        self.boundingBox.center = position
        self.mutable_bb = copy.deepcopy(self.boundingBox)

        self.speed = initial_speed
        self.last_heading = 0

    def move(self, offset):

        x, y = offset[0], offset[1]
        
        self.mutable_bb = self.mutable_img.get_rect(center = self.mutable_bb.center)

        self.mutable_bb.move_ip([x, y])


class Agent(Obstacles):
    def __init__(self, image_name, position, init_speed):
        super(self.__class__, self).__init__(image_name, position, init_speed)
        self.img = pygame.transform.scale(self.img, (config.car_width* 10, config.car_length* 10))
        self.mutable_img = self.img.copy()

        self.boundingBox = self.img.get_rect()
        self.boundingBox.center = position
        self.mutable_bb = copy.deepcopy(self.boundingBox)

    def move(self, offset):

        x, y = offset[0], offset[1]
        
        self.mutable_bb = self.mutable_img.get_rect(center=self.mutable_bb.center)

        self.mutable_bb.move_ip([x, y])


class Environment(object):
    def __init__(self, lane_list, obj_list, agent_list, state):
        
        self.lane_list = lane_list
        self.obj_list = obj_list
        self.agent_list = agent_list
        
        self.state = state
        
        self._renderFlag = False
        self._renderObj()

        self.maxAcc = 1
        self.distance = 0
        self.maxAngle = 90

    def _renderObj(self):
        displaySurface.fill((154, 205, 50))

        for bl in self.lane_list[0].bushList:
            displaySurface.blit(bl[0], bl[1])

        for br in self.lane_list[1].bushList:
            displaySurface.blit(br[0], br[1])

        for l in self.lane_list:
            pygame.draw.rect(displaySurface, l.color, l.lane_rect)

        for i in range(0, len(self.lane_list)-1):
            for m in self.lane_list[i].markerList:
                pygame.draw.rect(displaySurface, [255, 255, 255], m)

        for a in self.agent_list:
            displaySurface.blit(a.mutable_img, a.mutable_bb)

        for o in self.obj_list:
            displaySurface.blit(o.img, o.mutable_bb)

        pygame.display.update()
    
    def step(self, state, prev_state):
        global count
        pos = utils.state_to_plot(state)
        

        # next_speed, next_heading, lane_angle, lane_number, all_lane distance from center, off_road, opp_laneNum, opp_dist 
        for l in self.lane_list:
            l.relative_move([0, (state.x[0] - prev_state.x[0])*config.plot_to_real_ratio*2])
        
        # lane changing
        
        if(state.lane[1] == 0 and self.agent_list[1].mutable_bb.center[0] != config.left_lane_center):
            self.agent_list[1].move([-1, 0])
        
        # move human
        self.agent_list[1].move([0, pos[1] - self.agent_list[1].mutable_bb.center[1]])
        
        print(pos[1] - self.agent_list[1].mutable_bb.center[1])
        
        i = 2
        for o in self.obj_list:
            o.move([0, pos[i] - o.mutable_bb.center[1]])
            i += 1
            
        self._renderObj()
        
        pygame.image.save(displaySurface, "./figures/influence_velocity_{}.jpg".format(count))
        
        count += 1

        return
