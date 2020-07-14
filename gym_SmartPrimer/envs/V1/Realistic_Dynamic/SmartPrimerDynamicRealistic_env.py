import gym
from gym import spaces
from gym_SmartPrimer.envs.Realistic_Dynamic.ChildClassDynamic  import Child
import numpy as np
from gym_SmartPrimer.envs.Realistic_Dynamic import ChildBehaviorDynamic as ChildBehavior
import matplotlib.pyplot as plt
from gym_SmartPrimer.envs.Realistic_Dynamic import BehaviorNextObservation as NextObs

class SmartPrimerDynamicEnv(gym.Env):
	""" Realistic and dynamic Smart primer environment that simulates children trying to solve a geometry problem
	 using a simulation that samples from different simulations"""

	metadata = {'render.modes': ['human']}

	def __init__(self):
		self.env = {}
		self.info = {}
		#self.i = 0
		self.hints = [
		[0, 1, 2, 3],
		[0, 1, 3],
		[0, 2, 3],
		[1, 2, 3],
		[0, 3],
		[1, 3],
		[2, 3],
		[3],
		[0, 1, 2],
		[0, 1],
		[0, 2],
		[1, 2],
		[0],
		[1],
		[2]
		]
		self.childRewards = []
		self.RewardsPerChild = []
		self.performance = []
		self.perfect = []
		self.avgPerfect = []

		self.Ncorrect = []
		self.avgCorrect = []

		self.Nquit = []
		self.avgQuit = []

		self.Nearly = []
		self.avgEarly = []

		self.nHints = len(self.hints)

		low = np.array((0,-2,-2,-2,-2,0,0,0), dtype=float) #pre-test, 4 words dim, 3 prev-hints
		high = np.array((9,2,2,2,2,1,1,1), dtype=float) #pre-test, 4 words dim, 3 prev-hints

		self.observation_space = spaces.Box(low, high, dtype=np.float)
		self.action_space = spaces.Discrete(4)  # action space at start is all actions, but this actions space changes
		self.reward_range = (-2, 1)

		self.reset()

	def step(self, action):
		self.prev_hint = action
		prevNeededHint = self.child.neededHint

		if self.simulationSetting == 0:
			reward, done, info = ChildBehavior.react2hint1(action, self.child)
		elif self.simulationSetting == 1:
			reward, done, info = ChildBehavior.react2hint2(action, self.child)
		elif self.simulationSetting == 2:
			reward, done, info = ChildBehavior.react2hint3(action, self.child)
		elif self.simulationSetting == 3:
			reward, done, info = ChildBehavior.react2hint4(action, self.child)

		if not done:
			#self.state = NextObs.nextObservation1(self.child)
			self.state = NextObs.nextObservation2(self.child, action, prevNeededHint)

		# some performance tracking
		self.childRewards.append(reward)
		if done:
			self.RewardsPerChild.append(sum(self.childRewards))
			performance = self.RewardsPerChild[-min(len(self.RewardsPerChild), 100):]
			self.performance.append(np.mean(performance))

			self.perfect.append(info['perfect'])
			avgPerfect = np.mean(self.perfect[-min(len(self.RewardsPerChild), 100):])
			self.avgPerfect.append(avgPerfect)

			Nquit, Ncorrect, Nearly = 0, 0, 0
			if info['reaction'] == 'quit':
				Nquit = 1
			elif info['reaction'] == 'correct':
				Ncorrect = 1
			elif info['reaction'] == 'early':
				Nearly = 1

			self.Nquit.append(Nquit)
			avgQuit = np.mean(self.Nquit[-min(len(self.Nquit), 100):])
			self.avgQuit.append(avgQuit)

			self.Ncorrect.append(Ncorrect)
			avgCorrect = np.mean(self.Ncorrect[-min(len(self.Ncorrect), 100):])
			self.avgCorrect.append(avgCorrect)

			self.Nearly.append(Nearly)
			avgEarly = np.mean(self.Nearly[-min(len(self.Nearly), 100):])
			self.avgEarly.append(avgEarly)

		self.info = {'RewardsPerChild': self.RewardsPerChild, 'Performance': self.performance,
		             'Perfect': self.avgPerfect, 'Ncorrect': self.avgCorrect,
		             'Nquit': self.avgQuit, 'Nearly': self.avgEarly}

		return self.state, reward, done, self.info


	def reset(self):
		self.simulationSetting = np.random.randint(0, 4)
		#self.simulationSetting = 3

		self.child = Child(np.random.randint(0, self.nHints), self.hints)  # create a child of random type
		self.childRewards = []

		#self.state = NextObs.nextObservation1(self.child)
		self.state = NextObs.nextObservation2(self.child, -1, self.child.neededHint)

		return self.state

	def render(self, mode='human'):
		plt.plot(self.info['Performance'])
		plt.show()


