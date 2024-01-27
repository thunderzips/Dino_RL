# import math
# import random
# from collections import namedtuple, deque
# from itertools import count
# import time

# from environment import PlayField

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# import copy
# import numpy as np

# env = PlayField()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device = torch.device('cpu')

# class DQN(nn.Module):

#     def __init__(self, n_observations, n_actions):
#         super(DQN, self).__init__()
#         self.layer1 = nn.Linear(1508, 512)
#         self.layer2 = nn.Linear(512, 256)
#         self.layer3 = nn.Linear(256, 128)
#         self.layer4 = nn.Linear(128, 128)
#         self.layer5 = nn.Linear(128, 64)
#         self.layer6 = nn.Linear(64, 32)
#         self.layer7 = nn.Linear(32, n_actions)

#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=5)
#         self.conv2 = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=3)
#         self.flatten = nn.Flatten()
    
#     def forward(self, x):

#         try:
#             x = torch.reshape(x,(1,32,64))
#         except:
#             x = torch.reshape(x,(BATCH_SIZE,1,32,64))

#         x = self.conv1(x)
#         x = self.conv2(x)

#         x = self.flatten(x)
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         x = F.relu(self.layer3(x))
#         x = F.relu(self.layer4(x))
#         x = F.relu(self.layer5(x))
#         x = F.relu(self.layer6(x))
#         x = F.relu(self.layer7(x))
#         return x

# n_actions = 3
# state, info = env.reset()
# n_observations = env.state.shape[0]*env.state.shape[1]*env.state.shape[2]

# policy_net = DQN(n_observations, n_actions).to(device)
# policy_net.load_state_dict(torch.load("policy_net.pth"))
# steps_done = 0

# episode_durations = []

# if __name__=='__main__':

#     time.sleep(3)

#     for i_episode in range(10):
#         print(i_episode)
#         state, info = env.reset()
#         state = torch.tensor(state, dtype=torch.float32, device=device).flatten()#.unsqueeze(0)

#         for t in count():
#             chosen_action = policy_net(state)
#             action_ = torch.tensor([torch.argmax(chosen_action).item()], device=device, dtype=torch.long)
#             action = action_[0]

#             # print(i_episode, action)
#             observation, reward, done, _ = env.step(action)
#             reward = torch.tensor([reward], device=device)

#             if done:
#                 next_state = torch.zeros(observation.shape, device=device).flatten()#None
#             else:
#                 next_state = torch.tensor(observation, dtype=torch.float32, device=device).flatten()#unsqueeze(0)

#             state = next_state

#             if done:
#                 # print('done!!')
#                 episode_durations.append(t + 1)
#                 print("total reward = ",reward)
#                 env.reset()
#                 break

#     print('Complete')


