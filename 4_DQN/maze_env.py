import numpy as np
import time
import sys
import tkinter as tk

UNIT = 40 # 像素
MAZE_H = 4 # maze4格高
MAZE_W = 4 # maze4格宽

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT,  MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H*UNIT,
                                width=MAZE_W*UNIT)
        # 画格子    
        for c in range(0, MAZE_W*UNIT, UNIT):
            x0 = c
            y0 = 0
            x1 = c
            y1 = MAZE_H*UNIT
            self.canvas.create_line(x0,y0,x1,y1)
        for r in range(0, MAZE_H*UNIT, UNIT):
            x0 = 0
            y0 = r
            x1 = MAZE_W*UNIT
            y1 = r
            self.canvas.create_line(x0,y0,x1,y1)

        # 起点  
        origin = np.array([20,20])

        # 陷阱  
        # 左下那个
        hell1_center = origin + np.array([UNIT*2,UNIT]) # 100,60
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0]-15, hell1_center[1]-15, #第一对xy
            hell1_center[0]+15, hell1_center[1]+15,
            fill='purple'
        )

        # 右边那个
        # hell2_center = origin + np.array([UNIT, UNIT * 2])
        # self.hell2 = self.canvas.create_rectangle(
        #     hell2_center[0] - 15, hell2_center[1] - 15,
        #     hell2_center[0] + 15, hell2_center[1] + 15,
        #     fill='black')

        # 终点
        oval_center = origin + np.array([UNIT*2,UNIT*2])
        self.oval = self.canvas.create_oval(
            oval_center[0]-15, oval_center[1]-15,
            oval_center[0]+15, oval_center[1]+15,
            fill='yellow'
        )

        # red rect
        self.rect = self.canvas.create_rectangle(
            origin[0]-15, origin[1]-15,
            origin[0]+15, origin[1]+15,
            fill='red'
        )

        self.canvas.pack()

    def reset(self):  # red rect回到起点
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.rect)
        origin = np.array([20,20])
        self.rect = self.canvas.create_rectangle(
            origin[0]-15, origin[1]-15,
            origin[0]+15, origin[1]+15,
            fill='red'
        )
        # 返回状态
        return (np.array(self.canvas.coords(self.rect)[:2])-np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
    
    def step(self, action):
        s= self.canvas.coords(self.rect)
        base_action = np.array([0,0])
        if action == 0: #up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action ==1:
            if s[1] < (MAZE_H-1)*UNIT:
                base_action[1] += UNIT
        elif action == 2: # right
            if s[0] < (MAZE_W-1)*UNIT:
                base_action[0] += UNIT
        elif action == 3: #left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        
        self.canvas.move(self.rect, base_action[0], base_action[1])

        next_coords = self.canvas.coords(self.rect)

        # reward
        if next_coords == self.canvas.coords(self.oval):
            reward = 1
            done = True
        elif next_coords in [self.canvas.coords(self.hell1)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
        return s_, reward, done
    
    def render(self):
        # time.sleep(0.01)
        self.update()
