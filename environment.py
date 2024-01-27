import gym
from gym import spaces
import numpy as np
import copy
import pyautogui
import matplotlib.pyplot as plt
import cv2
import time

'''
PLAN:
    action space = up, down
    observation space = screen view 128x64
    reward = 1 for every time step

    done = 1 if game over

'''


class PlayField(gym.Env):
    def __init__(self):
        super(PlayField, self).__init__()

        self.action_space = spaces.Discrete(3)
                    
        self.steps = 0
        self.reward = 0

        self.reset()

    def reset(self):

        self.state = self.get_screenshot()
        self.check_game_over()
        self.resize_screenshot()

        self.steps = 0

        self.info = None
        self.game_over = False
        self.reward = 0

        time.sleep(1)
        self.start_time = time.time()

        return self.state, self.info

    def step(self, action):

        if action==0:
            pass
        elif action==1:
            pyautogui.hotkey('up')
        elif action==2:
            pass

        self.state = self.get_screenshot()
        self.check_game_over()
        self.resize_screenshot()
        self.compute_reward()
        done = self.is_done()

        self.steps += 1
        info = {}

        return self.state, self.reward, done, info

    def compute_reward(self):
        self.reward += 1

    def is_done(self):
        if self.game_over:
            print("done")
            return True
        else:
            return False

    def get_screenshot(self):
        screenshot = pyautogui.screenshot(region=(350,450, 1920-750, 1080//2-50))
        screenshot = np.array(screenshot)
        return screenshot
    
    def resize_screenshot(self):
        self.state = cv2.cvtColor(self.state, cv2.COLOR_BGR2GRAY)
        self.state = cv2.resize(self.state,(64,32)).reshape((1,32,64))

    def check_game_over(self):
        game_over_img = copy.copy(self.state)[:40, 400:800]

        if np.mean(game_over_img) > 235:
            self.game_over = False
        else:
            self.game_over = True