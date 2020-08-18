import numpy as np
import os
import json

def react2action(action, child, stage, interactions):
	'''Function that takes as input the action taken, the child object, problem stage and interactions, and returns
	the corresponding reward, whether the child is completely done and some information statistics'''

	#initialize
	done = False
	info = {}
	reward = 0

	with open(os.path.abspath(os.path.join(os.path.dirname(__file__), 'childConfig.json'))) as config_file:
		settings = json.load(config_file)

	#if the action is question, we increase the number of questions asked
	if action == 'question':
		child.nQuestions += 1

	#if the action is encourage, we decrease the number of encouragements needed by 1. If this makes it 0 and
	#number of hints needed is 0, we change the time needed to not be infinite.
	elif action == 'encourage':
		child.nEncouragements += 1
		child.encouragementsNeeded -= 1
		if child.neededHints <= 0 and child.encouragementsNeeded <= 0 and child.neededTime == np.inf:
			reward = 0.01
			child.neededTime = max(settings['meanNeededTimeAfterHints'][child.type] + round(np.random.normal(0, 1)), 0)

	# if the action is hint, we decrease the number of hints needed by 1. If this makes it 0 and
	# number of encouragements needed is 0, we change the time needed to not be infinite.
	elif action == 'hint':
		child.neededHints -= 1
		if child.neededHints <= 0 and child.encouragementsNeeded <= 0 and child.neededTime == np.inf:
			reward = 0.01
			child.neededTime = max(settings['meanNeededTimeAfterHints'][child.type] + round(np.random.normal(0, 1)), 0)
		elif child.neededHints < 0:
			child.nWrongHints += 1

	#define the quitting probability
	quitProb = max(0, interactions[1] - 120) * 0.0002 + child.nWrongAnswers * 0.01

	#if we encouraged, the probability is 0 for the first 3 encouragements
	if action == 'encourage' and child.nEncouragements <= 3:
		quitProb = 0
		reward = 0.01

	if action == 'hint' and child.neededHints > 0:
		quitProb -= 0.05
		reward = 0.01

	#clip the probability to make between 0 and 1
	quitProb = min(1, max(0, quitProb))

	# we are at the last question and the child does not need any more time (e.g. the child finishes)
	if stage == 3 and child.neededTime <= 0:
		reward = makePostTest(child)
		info['reaction'] = 'finished'
		return reward, True, info

	#if the child quits
	if np.random.binomial(1, quitProb) == 1:
		postResult = makePostTest(child)
		reward = -2
		info['reaction'] = 'quit'
		return reward, True, info

	return reward, done, info


def makePostTest(child):
	'''A function that takes as input the child and returns its post-test score - pre-test'''
	potential = 8 - child.preScore
	multiplier = min(1, max(0, 0.5 - child.nWrongHints * 0.1 + child.nQuestions * 0.07 + child.nEncouragements * 0.03))

	postImp = multiplier * potential
	return postImp