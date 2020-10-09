import gym
from gym import spaces
from gym_SmartPrimer.envs.V2.Realistic.Childclass  import Child
import numpy as np
from gym_SmartPrimer.envs.V2.Realistic import ChildBehavior as ChildBehavior
import matplotlib.pyplot as plt
from gym_SmartPrimer.envs.V2.Realistic import NextObservation as nextObs
import json
import os
import copy

class TestEnv(gym.Env):
	"""V2 smart primer environment"""

	metadata = {'render.modes': ['human']}

	def __init__(self):
		'''Initializes the environment'''
		# low = np.array((0, 2, 0, 0, 0, 0, 0, 0), dtype=float)
		# high = np.array((8, 6, 1, 1, 1, 3, 45, 10), dtype=float)  # pre-test, grade, 4 words dim, stage, anxiety ,number of wrong answers

		low = np.array((-1, -1, -1, -1, -1, -1, -1, -1), dtype=float)
		high = np.array((1, 1, 1, 1, 1, 1, 1, 1), dtype=float)

		self.observation_space = spaces.Box(low, high, dtype=np.float)

		self.action_space = spaces.Discrete(4)  #do nothing, encourage, ask question or provide hints
		self.actions = ['nothing', 'encourage', 'question',  'hint']
		self.reward_range = (-8, 9)
		self.env = {}


		self.childRewards = []
		self.plotRewards = []
		self.reset()

	def step(self, action):
		'''Takes an action and return the new statespace, reward, whether the episode has ended and some performance info'''

		action = self.actions[action]
		done = False
		# print(self.state)
		# print(self.state2)
		if (self.state == np.array([0,0,0,0,0,0,0,0])).all() and action == 'nothing':
			reward = 9
			self.state = [-9, 1, 0, 0, 0, 3, 0, 0]
			self.state = np.array(self.state, dtype=np.float)
			self.state2 = copy.deepcopy(self.state)

			# self.state = (self.state - [-4.5, 0, -5, 0, 0, 1.5, 25, 0]) / [9, 1, 10, 1, 1, 3, 50, 1]

		elif (self.state == np.array([-9, 1, 0, 0, 0, 3, 0, 0])).all() and action == 'encourage':
			reward = 9

			self.state = [0, 0, 1, -1, 1, 0, 50, 1]
			self.state = np.array(self.state, dtype=np.float)
			self.state2 = copy.deepcopy(self.state)

			# self.state = (self.state - [-4.5, 0, -5, 0, 0, 1.5, 25, 0]) / [9, 1, 10, 1, 1, 3, 50, 1]

		elif (self.state == np.array([0, 0, 1, -1, 1, 0, 50, 1])).all() and action == 'hint':
			reward = 9
			self.state = [1, -1, -10, -1, 0, 1, 46, -1]
			self.state = np.array(self.state, dtype=np.float)
			self.state2 = copy.deepcopy(self.state)

			# self.state = (self.state - [-4.5, 0, -5, 0, 0, 1.5, 25, 0]) / [9, 1, 10, 1, 1, 3, 50, 1]

		elif (self.state == np.array([1, -1, -10, -1, 0, 1, 46, -1])).all() and action == 'question':
			reward = 9

			self.state = [-1,-1,-1,1,1,1,1,1]
			self.state = np.array(self.state, dtype=np.float)
			self.state2 = copy.deepcopy(self.state)

			# self.state = (self.state - [-4.5, 0, -5, 0, 0, 1.5, 25, 0]) / [9, 1, 10, 1, 1, 3, 50, 1]

			done = True
			self.childRewards.append(self.rewards)
			self.plotRewards.append(np.mean(self.childRewards[- min(20, len(self.childRewards)):]))

		else:
			reward = -8

		self.rewards += reward


		return self.state, reward, done, self.rewards


	def reset(self):
		'''Starts a new episode by creating a new child and resetting performance, stage, observation space.'''

		self.rewards = 0
		self.state = [0,0,0,0,0,0,0,0]

		self.state = np.array(self.state, dtype=np.float)
		self.state2 = np.array([0,0,0,0,0,0,0,0], dtype=np.float)
		return self.state

	def render(self, mode='human'):
		plt.plot(self.plotRewards)
		plt.show()


