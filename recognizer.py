# Neural Net Optical Character Recognizer

# This python script first trains the neural net based on a number of sample digits and then 
# creates a window that the user can use to try writing digits to see if it can recognize them.

import math
import random
import time
import sys
import time
import copy
import pygame
from pygame.locals import *

import neuralnet2
import sample

try:
    maxcount = int(sys.argv[1])
except:
    maxcount = 50

rnd = random.Random(int(round(time.time() * 1000)))

n = neuralnet2.net(301,40,10,2)

training_set = []

for num,numset  in enumerate(sample.training_set):
    for x in numset:
        input_vector = [1]
        #desired_outputs = [0 for i in range(10)]
        desired_outputs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        desired_outputs[num] = 1
        for row in x:
            for pixel in row:
                input_vector.append(pixel)
        training_set.append((input_vector, desired_outputs))

random.shuffle(training_set)

count = 0

while True:
    total_error = 0
    error_count = 0
    for input_vector, desired_outputs in training_set: # loop through all of the input vectors 
        errors = n.train(input_vector, desired_outputs)
        total_error += sum(error * error for error in errors) # we are trying to minimize the total error.
        for error in errors:
            if abs(error) > .4:
                error_count += 1
    if count > maxcount:
        break
    if count == 0:
        print('count, error count, total error:')
    #if count % 10 == 0:
    print (count, error_count, total_error) # print out the current total weight to provide a visual for how the net is learning
        #print (error_count)
    count = count + 1 
#for input_vector, desired_output in training_set: # loop through all of the input vectors 
    #print (n.out(input_vector)[1], desired_output, input_vector)

#--------------pygame

SCREEN_SIZE = [300, 400]
W = SCREEN_SIZE[0]
H = SCREEN_SIZE[1]
SCREEN = pygame.display.set_mode(SCREEN_SIZE)
TIME   = pygame.time
CLOCK  = pygame.time.Clock()
BLACK  = [0,0,0]
RED    = [255, 0, 0]
BLUE   = [0, 0, 40]
GREEN  = [0, 255, 0]
GRAY   = [100, 100, 100]
TOP    = [SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] // 10]

HEIGHT = 20
WIDTH = 15
empty = [[0 for x in range(WIDTH)] for x in range(HEIGHT)]

def new_board(h, w):
    board = [[0]*w for i in range(h)]
    #b0 = []
    #for i in range(20):
        #for j in range(15):
            #board[i][j] = b0[i*15+j]
    return board

done = False
class World: # mostly a variable holder maybe more?
    def __init__(self):
        self.board=new_board(HEIGHT, WIDTH)
        self.board = copy.deepcopy(empty)
        self.h = len(self.board)
        self.w = len(self.board[0])
        self.h_ratio = H // self.h //2
        self.w_ratio = W // self.w //2
        self.rect_width = W // self.w
        self.rect_height = H // self.h
        self.click = pygame.mouse.get_pressed()
        self.mid = False
        self.time = 10
        self.frames = 0
        self.pause_wait = 0
        self.paused = 0
        self.saved = self.board
        self.rects = []
        self.draw_rects = []
        self.press_time = time.time()
        self.last_space = False
        pass

    def update(self):
        if self.paused:
            SCREEN.fill(BLUE)
            #get inputs / other things...
        else:
            SCREEN.fill(BLACK)
        pygame.event.pump()
        self.mousexy = pygame.mouse.get_pos()
        self.keystate = pygame.key.get_pressed()
        self.cellx = self.mousexy[0] // self.rect_width
        self.celly = self.mousexy[1] // self.rect_height
        if not self.last_space and self.keystate[K_SPACE]:
            input_vector = [1]
            for row in self.board:
                for pixel in row:
                    input_vector.append(pixel)
            outs = n.out(input_vector)[1]
            printout = []
            for i, out in enumerate(outs):
                printout.append((i, out))
            for x in sorted(printout, key=lambda l:l[1], reverse=True):
                print(x)
            print('-----\n\n')
        self.last_space = self.keystate[K_SPACE]
        if pygame.mouse.get_pressed()[0]:
            self.board[self.celly][self.cellx] = 1
        if pygame.mouse.get_pressed()[2]:
            self.board[self.celly][self.cellx] = 0
        if pygame.mouse.get_pressed()[1] and time.time() - self.press_time >= 0.5:
            self.startxy = [self.cellx, self.celly]
            self.press_time = time.time()
            self.mid = not self.mid
        for y, row in enumerate(self.board):
            for x, val in enumerate(row):
                if val:
                    self.left = x * self.rect_width
                    self.top = y * self.rect_height
                    self.rects.append(pygame.Rect(self.left, self.top, self.rect_width, self.rect_height))
        for r in self.rects:
            pygame.draw.rect(SCREEN,GRAY, r)
        self.rects = []
        self.draw_rects = []
        self.frames+=1
        pygame.display.flip()
        CLOCK.tick(60)

pygame.init()

world = World()
while not done:
    world.update()
pygame.quit()
input('\nPress Enter To Quit')
sys.exit()
