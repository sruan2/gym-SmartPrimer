import numpy as np
import os
import json

def react2action(action, child, stage, interactions):
	'''Function that takes as input the action taken, the child object, problem stage and interactions, and returns
	the corresponding reward, whether the child is completely done and some information statistics'''

	pureImprovement = True
	perfectActions = False

	if perfectActions:
		if child.neededHints>0:
			action = 'hint'
		elif child.encouragementsNeeded>0:
			action = 'encourage'
		else:
			action = 'question'


	#initialize
	info = {}
	reward = 1
	wrongHint = 0

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
			child.neededTime = max(settings['meanNeededTimeAfterHints'][child.type] + round(np.random.normal(0, 1)), 0)

	# if the action is hint, we decrease the number of hints needed by 1. If this makes it 0 and
	# number of encouragements needed is 0, we change the time needed to not be infinite.


	elif action == 'hint':
		child.neededHints -= 1
		if child.neededHints <= 0 and child.encouragementsNeeded <= 0 and child.neededTime == np.inf:
			child.neededTime = max(settings['meanNeededTimeAfterHints'][child.type] + round(np.random.normal(0, 1)), 0)
		elif child.neededHints < 0:
			wrongHint = 1
			child.nWrongHints += 1
		child.nHints += 1

	#define the quitting probability
	# quitProb = max(0, interactions[1] - 120) * 0.0002 + child.nWrongAnswers * 0.01
	quitProb = max(0, interactions[1] - 120) * 0.0002 + child.nWrongAnswers * 0.003 + wrongHint * 0.01

	#if we encouraged, the probability is 0 for the first 3 encouragements
	if action == 'encourage' and child.nEncouragements <= 3:
		reward = 1.5
		quitProb -= 0.05

	if action == 'hint' and child.neededHints > 0:
		child.nCorrectHints += 1
		reward = 1.5
		quitProb -= 0.05

	#clip the probability to make between 0 and 1
	quitProb = min(1, max(0, quitProb))

	# we are at the last question and the child does not need any more time (e.g. the child finishes)
	if stage == 3 and child.neededTime <= 0:
		improvement = makePostTest(child)
		reward = 1 + max(0, improvement) - (1 + child.psi) * (child.nHints + child.nEncouragements + child.nQuestions)

		info['reaction'] = 'finished'

		if pureImprovement:
			reward = improvement
		return improvement, reward, True, info

	#if the child quits
	if np.random.binomial(1, quitProb) == 1:
		improvement = 0
		# postResult = makePostTest(child)
		reward = - 8 - (1 + child.psi) * (child.nHints + child.nEncouragements + child.nQuestions)
		info['reaction'] = 'quit'

		if pureImprovement:
			reward = - 8

		return improvement, reward, True, info

	# if reward != 2:
	# 	reward = 1
	if action == 'nothing':
		reward = 0

	if pureImprovement:
		reward = 0

	improvement = makePostTest(child)
	return improvement, reward, False, info


def makePostTest(child):
	'''A function that takes as input the child and returns its post-test score - pre-test'''
	potential = 8 - child.preScore
	#multiplier = min(1, max(0, 0.5 - child.nWrongHints * 0.1 + child.nQuestions * 0.07 + child.nEncouragements * 0.03))

	# print('Number of wrong hints: {}'.format(child.nWrongHints))
	# print('Number of questions: {}'.format(child.nQuestions))
	# print('Number of encouragements: {}'.format(child.nEncouragements))
	# print('Number of wrong answers: {}'.format(child.nWrongAnswers))
	# print('Number of correct hints: {}'.format(child.nCorrectHints))

	nCorrectHints = min(child.nCorrectHints, 4)
	multiplier = min(1, max(0, 0.7 - child.nWrongHints * 0.1 + child.nQuestions * 0.07 + child.nEncouragements * 0.05 - child.nWrongAnswers * 0.05 + nCorrectHints * 0.08))

	postImp = multiplier * potential
	return postImp