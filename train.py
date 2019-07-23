#!/usr/bin/env python3

import numpy as np
import argparse
from copy import deepcopy
import torch
import torch.nn as nn
import gym

from normalized_env import NormalizedEnv
from evaluator import Evaluator
from ddpg_model import Actor, Critic
from util import *
from torch.optim import Adam
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess


'''
    All hyperparameters
'''
# Enviroment
ENVIRONMENT = "Pendulum-v0"
DISCOUNT = 0.99
# OU process
OU_THETA = 0.15
OU_SIGMA = 0.2
OU_MU = 0.0
# Set up memory
MEMORY_SIZE = 6000000
# Training Procedure
LEARNING_RATE = 1e-3
MAX_EPISODE = 500
MAX_STEP_PER_EPISODE = 400
DELTA_EPSILON = 1.0/50000
TAU = 0.001
CRITERION = nn.MSELoss()
BATCH_SIZE = 64
# Evaluate Procedure
MAX_EPISODE_EVA = 20
MAX_STEP_PER_EPISODE_EVA = 500
EVALUATING_EPISODE_INTERVAL = 50
LOG_PATH = "log"
OPEN_VISUALIZATION_EVA = True
USE_CUDA = torch.cuda.is_available()
# A class for train DDPG
class DDPG_trainer(object):
    def __init__(self, nb_state, nb_action):
        self.nb_state = nb_state
        self.nb_action = nb_action

        self.actor = Actor(self.nb_state, self.nb_action)
        self.actor_target = Actor(self.nb_state, self.nb_action)
        self.actor_optim  = Adam(self.actor.parameters(), lr=LEARNING_RATE)

        self.critic = Critic(self.nb_state, self.nb_action)
        self.critic_target = Critic(self.nb_state, self.nb_action)
        self.critic_optim  = Adam(self.critic.parameters(), lr=LEARNING_RATE)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        #Create replay buffer
        self.memory = SequentialMemory(limit=MEMORY_SIZE, window_length=1)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_action, theta=OU_THETA, mu=OU_MU, sigma=OU_SIGMA)

        self.is_training = True
        self.epsilon = 1.0
        self.a_t = None
        self.s_t = None

        if USE_CUDA: self.cuda()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def select_action(self, s_t, decay_epsilon=True):

        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)
        action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= DELTA_EPSILON

        self.a_t = action
        return action

    def reset(self, observation):
        self.start_state = observation
        self.random_process.reset_states()

    def observe(self, r_t, s_t1, done):

        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def update_all(self):
        # Help Warm Up
        if self.memory.nb_entries < BATCH_SIZE*2:
            return

        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(BATCH_SIZE)

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target([
                to_tensor(next_state_batch),
                self.actor_target(to_tensor(next_state_batch)),
            ])

        target_q_batch = to_tensor(reward_batch) + \
                         DISCOUNT * to_tensor(terminal_batch.astype(np.float)) * next_q_values

        # Critic update
        self.critic.zero_grad()
        for state in state_batch:
            if state.shape[0]<=2:
                # print("Error sampled memory!")
                return

        q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])
        value_loss = CRITERION(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, TAU)
        soft_update(self.critic_target, self.critic, TAU)


# Main Training Algorithm
def train(trainer, env):

    evaluator = Evaluator(MAX_EPISODE_EVA, EVALUATING_EPISODE_INTERVAL * MAX_STEP_PER_EPISODE,
                  save_path=LOG_PATH, max_episode_length=MAX_STEP_PER_EPISODE_EVA)
    for episode in range(MAX_EPISODE):
        # Initialize random process
        observation = deepcopy(env.reset())
        trainer.reset(observation)
        step = 0
        episode_reward = 0
        done = False
        for step in range(MAX_STEP_PER_EPISODE):
            action = trainer.select_action(observation)
            observation2, reward, done, info = env.step(action)
            observation2 = deepcopy(observation2)
            if step >= MAX_STEP_PER_EPISODE - 1:
                done = True

            # trainer store transitions and update all networks
            trainer.observe(reward, observation2, done)
            trainer.update_all()
            episode_reward += reward
            observation = deepcopy(observation2)
        print('Training Episode {}, Episode Reward is:{}'.format(episode,episode_reward))
        if episode % EVALUATING_EPISODE_INTERVAL == 0:
            policy = lambda x: trainer.select_action(x, decay_epsilon=False)
            evaluator(env, policy, debug=True,visualize=OPEN_VISUALIZATION_EVA, save=True)

'''
    Entrance of Main Program
'''
env = NormalizedEnv(gym.make(ENVIRONMENT))
nb_states = env.observation_space.shape[0]
nb_actions = env.action_space.shape[0]

ddpg_trainer = DDPG_trainer(nb_states, nb_actions)
train(ddpg_trainer, env)