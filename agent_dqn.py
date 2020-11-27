#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch as T
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

T.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.gamma = args.gamma
        self.epsilon = args.eps
        self.lr = args.lr
        self.n_actions = env.action_space.n
        self.input_dims = (env.observation_space.shape)
        self.batch_size = args.batch_size
        self.eps_min = args.eps_min
        self.eps_dec = args.eps_dec
        self.replace_target_cnt = args.replace
        self.algo = args.algo
        self.mem_size= args.mem_size
        self.env_name = args.env_name
        self.chkpt_dir = args.chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.test = args.test_dqn
        self.cnt=0
        
        self.q_eval_name = 'BreakoutNoFrameskip-v4'+'_'+self.algo+'_q_eval.pth'
        self.q_next_name = 'BreakoutNoFrameskip-v4'+'_'+self.algo+'_q_next.pth'
        self.memory = ReplayBuffer(self.mem_size, self.input_dims, self.n_actions)
        
        self.q_eval = DQN(args,in_channels=4, num_actions=self.n_actions)
        self.q_next = DQN(args,in_channels=4, num_actions=self.n_actions)
        
        
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            self.load_checkpoint(self.q_eval,name=self.q_eval_name)
            self.load_checkpoint(self.q_next,name=self.q_next_name)

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.cnt+=1
        if not self.test:
            if np.random.random() > self.epsilon:
                state = np.array([observation], copy=False, dtype=np.float32)

                state_tensor = T.FloatTensor(state).to(self.q_eval.device)  
                _, advantages = self.q_eval.forward(state_tensor)

                action = T.argmax(advantages).item()
            else:
                action = np.random.choice(self.action_space)       
                       
        else:
            state = np.array([observation], copy=False, dtype=np.float32)

            state_tensor = T.FloatTensor(state).to(self.q_eval.device)  
            _, advantages = self.q_eval.forward(state_tensor)

            action = T.argmax(advantages).item()
        if test:
            if self.cnt%100==0:
                action=1
        return action
    
    def push(self,state, action, reward, state_, done):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.memory.store_transition(state, action, reward, state_, done)
        

        
        
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)
        
        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        
        return states, actions, rewards, states_, dones
         
        

    def train(self):
        """
        Implement your training algorithm here
        """
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.replay_buffer()
        indices = np.arange(self.batch_size)
        
        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s,
                        (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]

        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1,keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
        
    def load_models(self):
        self.load_checkpoint(self.q_eval,name=self.q_eval_name)
        self.load_checkpoint(self.q_next,name=self.q_next_name)
        
    def save_models(self):
        self.save_checkpoint(self.q_eval,name=self.q_eval_name)
        self.save_checkpoint(self.q_next,name=self.q_next_name)
        
    def save_checkpoint(self,model,name):
        print('... saving checkpoint ...')
        checkpoint_file = os.path.join(self.chkpt_dir, name)
        T.save(model.state_dict(), checkpoint_file)
        T.save({            
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict()

            }, checkpoint_file)

    def load_checkpoint(self,model,name):
        print('... loading checkpoint ...')
        checkpoint_file = os.path.join(self.chkpt_dir, name)
        model.load_state_dict(T.load(checkpoint_file)['model_state_dict'])
        model.optimizer.load_state_dict(T.load(checkpoint_file)['optimizer_state_dict'])        
    
    def replace_target_network(self):
        if self.replace_target_cnt is not None and \
           self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min
        
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.uint8)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.uint8)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        

        return states, actions, rewards, states_, terminal
    

