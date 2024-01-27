import math
import random
from collections import namedtuple, deque
from itertools import count
import time

from environment import PlayField

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import copy
import numpy as np
import cv2
import pyautogui
env = PlayField()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(1508, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 128)
        self.layer5 = nn.Linear(128, 64)
        self.layer6 = nn.Linear(64, 32)
        self.layer7 = nn.Linear(32, n_actions)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=3)
        self.flatten = nn.Flatten()
    
    def forward(self, x):

        try:
            x = torch.reshape(x,(1,32,64))
        except:
            x = torch.reshape(x,(BATCH_SIZE,1,32,64))

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        return x

BATCH_SIZE = 32
GAMMA = 0.7
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.05
LR = 0.0001

n_actions = 3
state, info = env.reset()
n_observations = env.state.shape[0]*env.state.shape[1]*env.state.shape[2]

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(100000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            chosen_action = policy_net(state)
            return torch.tensor([torch.argmax(chosen_action).item()], device=device, dtype=torch.long)
    else:
        return torch.tensor([env.action_space.sample()], device=device, dtype=torch.long)

episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state]).reshape((BATCH_SIZE,n_observations))
    
    action_batch = torch.cat(batch.action).reshape((BATCH_SIZE,1))
    state_batch = torch.cat(batch.state).reshape((BATCH_SIZE,n_observations))
    reward_batch = torch.cat(batch.reward).reshape((BATCH_SIZE,1))

    state_action_values = policy_net(state_batch).gather(1, action_batch.type(torch.int64))

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


num_episodes = 500

if __name__=='__main__':

    time.sleep(1)

    for i_episode in range(num_episodes):
        print(i_episode)
        state, info = env.reset()
        pyautogui.hotkey('up')

        state = torch.tensor(state, dtype=torch.float32, device=device).flatten()#.unsqueeze(0)

        for t in count():
            action_ = copy.copy(select_action(state))
            action = action_[0]

            observation, reward, done, _ = env.step(action)
            reward = torch.tensor([reward], device=device)

            if done:
                next_state = torch.zeros(observation.shape, device=device).flatten()#None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).flatten()#unsqueeze(0)

            memory.push(state, action_, next_state, reward)

            state = next_state

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                torch.save(policy_net.state_dict(),'policy_net.pth')
                print('saved,',"total reward = ",reward)
                break

    print('Complete')


