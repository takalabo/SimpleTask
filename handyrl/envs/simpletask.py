# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# implementation of simpletask

import copy
import random
from matplotlib.pyplot import axis

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from ..environment import BaseEnvironment


class Conv(nn.Module):
    def __init__(self, filters0, filters1, kernel_size, bn, bias=True):
        super().__init__()
        if bn:
            bias = False
        self.conv = nn.Conv2d(
            filters0, filters1, kernel_size,
            stride=1, padding=kernel_size//2, bias=bias
        )
        self.bn = nn.BatchNorm2d(filters1) if bn else None

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h


class Head(nn.Module):
    def __init__(self, input_size, out_filters, outputs):
        super().__init__()

        self.board_size = input_size[1] * input_size[2]
        self.out_filters = out_filters

        self.conv = Conv(input_size[0], out_filters, 1, bn=False)
        self.activation = nn.LeakyReLU(0.1)
        self.fc = nn.Linear(self.board_size * out_filters, outputs, bias=False)

    def forward(self, x):
        h = self.activation(self.conv(x))
        h = self.fc(h.view(-1, self.board_size * self.out_filters))
        return h

class SimpleModel(nn.Module):
    def __init__(self, hyperplane_n):
        super().__init__()
        self.relu = nn.ReLU()
        #100 ,256, 512 ,1024, 2048, 4096
        nn_size = 100
        self.fc1 = nn.Linear(hyperplane_n + 1, nn_size)
        self.head_p = nn.Linear(nn_size, 2**hyperplane_n)
        self.head_v = nn.Linear(nn_size, 1)
    def forward(self, x, hidden=None):
        h = F.relu(self.fc1(x))
        h_p = self.head_p(h)
        h_v = self.head_v(h)
        return {'policy': h_p, 'value': torch.tanh(h_v)}

# base class of Environment

class Environment(BaseEnvironment):
    def __init__(self, args={}):
        super().__init__()
        self.param = args['param']
        self.depth = self.param['depth']
        self.hyperplane_n = self.param['hyperplane_n']
        self.treasure = np.array(self.param['treasure'])
        self.set_reward = self.param['set_reward']
        self.start_random = self.param['start_random']
        self.pom_bool = self.param['pomdp_setting']['pom_bool']
        self.pom_state = self.param['pomdp_setting']['pom_state']
        self.pom_flag = 0
        self.tree_s = []
        self.action_list = []
        self.tree_make()

    def Transition(self, action, state):
        action = self.action_list_np[action]

        # print("state : ",state)
        # print("action : ",action)
        # print("state_depth : ", state)

        new_state_tmp = [state[i+1]+ action[0][i] for i in range(self.hyperplane_n)]
        new_state = (state[0]+1,) + tuple(new_state_tmp)

        return np.array(new_state)

    def tree_make(self):
        for i in range(2,2+self.depth):
            coord_seed = list(range(0,i))
            coord = list(itertools.product(coord_seed, repeat = self.hyperplane_n))
            coordinate = [(i-2,) + t for t in coord]
            self.tree_s += coordinate
            if i == 2:
                self.action_list = coord

        if not self.start_random:
            start = (-1,) + (0,) * self.hyperplane_n
            self.tree_s = np.insert(self.tree_s, 0, start, axis=0)

        self.tree_np = np.array(self.tree_s)
        self.action_list_np = np.array(self.action_list)

    def reset(self, args={}):
        if self.start_random:
            start = np.random.randint(len(self.action_list_np))
            self.state = self.tree_np[start]
        else:
            self.state = self.tree_np[0]

    def play(self, action, player):
        a = np.array([action], )
        next_s = self.Transition(a,self.state)
        self.state = next_s
        if self.pom_bool and np.array_equal(self.state, self.pom_state):
            self.pom_flag = 1

    def terminal(self):
        return self.state[0] == self.depth-1

    def outcome(self):
        outcomes = [-1]
        if self.pom_bool:
            if self.pom_flag and (self.state == self.treasure).all(axis=1).any():
                outcomes = [1]
            self.pom_flag = 0
        else:
            if (self.state == self.treasure).all(axis=1).any():
                outcomes = [1]
        return {p: outcomes[idx] for idx, p in enumerate(self.players())}

    def players(self):
        return [0]

    def net(self):
        return SimpleModel(self.hyperplane_n)

    def observation(self, player=None):
        return self.state.astype(np.float32)

    def legal_actions(self, player):
        legal_actions = np.arange(2**self.hyperplane_n)
        return legal_actions

if __name__ == '__main__':
    e = Environment()
    for _ in range(100):
        e.reset()
        while not e.terminal():
            print(e)
            e.play()
        print(e)
        print(e.outcome())
