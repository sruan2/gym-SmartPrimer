import gym
from gym import spaces
from gym_SmartPrimer.envs.V2.knownPersonas.Childclass  import Child
import numpy as np
from gym_SmartPrimer.envs.V2.knownPersonas import ChildBehavior as ChildBehavior
import matplotlib.pyplot as plt
from gym_SmartPrimer.envs.V2.knownPersonas import NextObservation as nextObs
import json
import os

class SmartPrimerKnownEnv(gym.Env):
	"""V2 smart primer environment"""

	metadata = {'render.modes': ['human']}

	def __init__(self):
		'''Initializes the environment'''
		with open(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'childConfig.json'))) as config_file:
			self.settings = json.load(config_file)

		self.env = {}
		self.info = {}

		self.RewardsPerChild = []
		self.performance = []

		self.nFinish = []
		self.avgFinish = []

		self.Nquit = []
		self.avgQuit = []

		self.childrenSimulated = 0

		# persona, seconds of last interaction, seconds of last correct answer, [0,0,0] (positive, idk, negative) words since last action taken, stage of the problem, seconds since last interaction with wizard
		low = np.array((0, 0, 0, 0, 0, 0, 0, 0), dtype = float)
		high = np.array((1, 1, 1, 1, 1, 1, 1, 3), dtype = float)

		self.observation_space = spaces.Box(low, high, dtype=np.float)

		self.action_space = spaces.Discrete(4)  #do nothing, encourage, ask question or provide hints
		self.actions = ['nothing', 'encourage', 'question', 'hint']
		self.reward_range = (-2, 10)

		self.actionInfo = {'nothing': [], 'encourage': [], 'question': [], 'hint': []}
		self.avgActionInfo = {'nothing': [], 'encourage': [], 'question': [], 'hint': []}

		kids = range(0, self.settings['nTypes'])
		for kid in kids:
			self.actionInfo[str(kid)] = [[], [], [], []]
			self.avgActionInfo[str(kid)] = [[], [], [], []]

		self.reset()

	def step(self, action):
		'''Takes an action and return the new statespace, reward, whether the episode has ended and some performance info'''
		self.getInfo(action)

		# the needed time for a child will decrease with one.
		self.child.neededTime -= 1
		action = self.actions[action]
		reward, done, info = ChildBehavior.react2action(action, self.child, self.stage, self.interactions)
		self.childRewards += reward

		if not done:
			self.state, self.interactions, self.stage = nextObs.nextObservation(self.child, self.interactions, action,
			                                                                    self.stage)  # first 30 secs

		if done:
			self.RewardsPerChild.append(self.childRewards)
			performance = self.RewardsPerChild[-min(len(self.RewardsPerChild), 100):]
			self.performance.append(np.mean(performance))

			nQuit, nFinish = 0, 0
			if info['reaction'] == 'quit':
				nQuit = 1
			elif info['reaction'] == 'finished':
				nFinish = 1

			self.Nquit.append(nQuit)
			avgQuit = np.mean(self.Nquit[-min(len(self.Nquit), 100):])
			self.avgQuit.append(avgQuit)

			self.nFinish.append(nFinish)
			avgFinish = np.mean(self.nFinish[-min(len(self.nFinish), 100):])
			self.avgFinish.append(avgFinish)

		self.info = {'RewardsPerChild': self.RewardsPerChild, 'Performance': self.performance, 'nFinish': self.avgFinish,
		             'nQuit': self.avgQuit, 'actionInfo': self.avgActionInfo}

		return self.state, reward, done, self.info

	def getInfo(self, actionTaken):
		'''Retreives performance measures for plotting'''
		kids = range(0, self.settings['nTypes'])
		actions = [0, 1, 2, 3]

		#kid specific
		for kid in kids:
			for action in actions:
				if kid == self.child.type and action == actionTaken:
					self.actionInfo[str(self.child.type)][actionTaken].append(1)
					temp = self.actionInfo[str(self.child.type)][actionTaken]
					self.avgActionInfo[str(self.child.type)][actionTaken].append(np.mean(temp[-min(len(temp), 500):]))
				elif kid == self.child.type:
					self.actionInfo[str(kid)][action].append(0)
					temp = self.actionInfo[str(kid)][action]
					self.avgActionInfo[str(kid)][action].append(np.mean(temp[-min(len(temp), 500):]))

		#for all kids
		for action in actions:
			if action == actionTaken:
				self.actionInfo[self.actions[action]].append(1)
			else:
				self.actionInfo[self.actions[action]].append(0)

			self.avgActionInfo[self.actions[action]].append(np.mean(self.actionInfo[self.actions[action]][-min(len(self.actionInfo[self.actions[action]]), 500):]))
		return 0


	def reset(self):
		'''Starts a new episode by creating a new child and resetting performance, stage, observation space.'''
		self.childrenSimulated += 1
		if self.childrenSimulated % 50 == 0:
			print('We simulated {} children'.format(self.childrenSimulated))

		self.interactions = [0, 0, 0]
		self.child = Child(self.settings)  # create a child of random type
		self.childRewards = 0

		prevAction = 'nothing'
		self.stage = 0

		self.state, self.interactions, self.stage = nextObs.nextObservation(self.child, self.interactions, prevAction, self.stage) #first 30 secs
		return self.state

	def render(self, mode='human'):
		'''Creates plots of the results'''
		ax1 = plt.subplot(311)
		# ax1.set_title('Scenario 4: average reward of last 100 children without quitting penalty')
		ax1.margins(0.05)
		# ax1.set_xlabel('Number of children')
		ax1.set_title('Average reward of last 100 children')
		ax1.plot(self.info['Performance'], 'r')

		ax4 = plt.subplot(312)
		ax4.set_title('Actions taken')
		ax4.plot(self.info['actionInfo']['question'], 'r', label='Question')
		ax4.plot(self.info['actionInfo']['encourage'], 'g', label='Encourage')
		ax4.plot(self.info['actionInfo']['hint'], 'b', label='Hint')
		ax4.plot(self.info['actionInfo']['nothing'], 'y', label='Nothing')
		ax4.set_ylabel('% of last 500 actions')
		ax4.legend()

		ax2 = plt.subplot(313)
		ax2.plot(self.info['nQuit'], 'r', label='Quit')
		ax2.plot(self.info['nFinish'], 'g', label='Finished')
		ax2.set_ylabel('% of last 100 children')
		ax2.legend()

		plt.show()

		nChildren = self.settings['nTypes']
		fig, axes = plt.subplots(nrows=1, ncols=nChildren)

		i = 0
		if nChildren != 1:
			for ax in axes.flatten():
				ax.set_title('Child type {}'.format(i + 1))
				ax.plot(self.info['actionInfo'][str(i)][0], 'y', label='Nothing')
				ax.plot(self.info['actionInfo'][str(i)][1], 'g', label='Encourage')
				ax.plot(self.info['actionInfo'][str(i)][2], 'r', label='Question')
				ax.plot(self.info['actionInfo'][str(i)][3], 'b', label='Hint')
				ax.set_ylabel('% of last 500 actions')
				ax.legend()
				i = +1
		else:
			axes.set_title('Child type {}'.format(i + 1))
			axes.plot(self.info['actionInfo'][str(i)][0], 'y', label='Nothing')
			axes.plot(self.info['actionInfo'][str(i)][1], 'g', label='Encourage')
			axes.plot(self.info['actionInfo'][str(i)][2], 'r', label='Question')
			axes.plot(self.info['actionInfo'][str(i)][3], 'b', label='Hint')
			axes.set_ylabel('% of last 500 actions')
			axes.legend()

		plt.show()


