import numpy as np
import json
import os

def usesWords(child, prevAction):
	'''Function that looks at previous action and current child state and returns the wordvector the child will produce'''
	words = [0, 0, 0] #postive, idk, negative

	if np.random.binomial(1, 0.8): #80% chance that the child uses the word corresponding to what it needs
		if prevAction == 'question':
			if child.neededHints > 0:
				words[2] = 1
			else:
				words[0] = 1
		elif prevAction == 'hint':
			if child.neededHints > 0:
				words[0] = 1
			else:
				words[1] = 1
		elif prevAction == 'nothing':
			words[1] = 1
		elif prevAction == 'encourage':
			words[0] = 1
	else:
		words[np.random.randint(0, 3)] = 1

	return words

def nextObservation(child, interactions, prevAction, stage):
	'''Function that looks at previous action and returns the next observation vector, along with the stage and new
	interaction times'''

	#get settings
	with open(os.path.abspath(os.path.join(os.path.dirname(__file__), 'childConfig.json'))) as config_file:
		info = json.load(config_file)

	#get let interaction with screen, correct answer and with wizard
	secLastScreen = interactions[0]
	secLastCor = interactions[1]
	secLastWiz = interactions[2]

	#init
	words = [0, 0, 0]
	question = 0
	hint = 0

	if prevAction == 'question':
		question = 1
	if prevAction == 'hint':
		hint = 1

	# if the child is ready to finish the sub-problem
	if child.neededTime <= 0:
		#he gives a correct answer in the last 30 secs
		secLastCor = np.random.randint(0, 30)

		#he interacted with the screen in the last 30 secs
		secLastScreen = secLastCor

		#stage will go up
		stage += 1

		#reset the number of hints a child needs
		child.neededHints = min(max(info['meanNeededHints'][child.type] + round(np.random.normal(0, 1)), 0), 4)

		child.encouragementsNeeded = max(0, info['meanNeededEncouragements'][child.type] + round(np.random.normal(0, 0.3)))

		#reset the timesteps the child has to wait for again
		if child.neededHints == 0 and child.encouragementsNeeded <= 0:
			child.neededTime = max(info['meanNeededTimeAfterHints'][child.type] + round(np.random.normal(0, 1)), 0)
		else:
			child.neededTime = np.inf

	#if the child is not able to finish the sub-problem
	else:
		#the probabilities for children to contact the wizard given that they need more hints
		wizProbsHintNeeded = np.array(info['probWordsHintNeeded']) + (0.3 * question) + (0.3 * hint)
		wizProbsNoHintNeeded = np.array(info['probWordsNoHintNeeded']) + (0.3 * question) + (0.3 * hint)

		#if the child does not need more hints
		if child.neededHints == 0:
			if np.random.binomial(1, min(1, wizProbsNoHintNeeded[child.type])) == 1:
				secLastWiz = np.random.randint(0, 30)

				words = usesWords(child, prevAction)

		#if the child needs more hints he might contact the wizard
		else:
			#if the child says something to the wizard
			if np.random.binomial(1, min(1, wizProbsHintNeeded[child.type])) == 1:
				secLastWiz = np.random.randint(0, 30)

				words = usesWords(child, prevAction)

				secLastScreen = secLastWiz

		#if the child was not able to finish the sub-problem, but attempted a wrong answer
		screenIntProbs = np.array(info['probWrongAnswer']) + interactions[0]*0.0005 #prob depends on how long he/she hasn't interacted with screen

		#if the child used the screen in the last 30 secs
		if np.random.binomial(1, screenIntProbs[child.type]) == 1:
			secLastScreen = min(secLastWiz, np.random.randint(0, 30))
			child.nWrongAnswers += 1

	#if the child did not interact with the screen
	if secLastScreen == interactions[0]:
		secLastScreen = interactions[0] + 30

	#if the child did give the correct answer
	if secLastCor == interactions[1]:
		secLastCor = interactions[1] + 30

	#if the child did not interact with the wizard
	if secLastWiz == interactions[2]:
		secLastWiz = interactions[2] + 30

	#update interactions variable
	interactions[0] = secLastScreen
	interactions[1] = secLastCor
	interactions[2] = secLastWiz

	persona = [0, 0, 0, 0]
	persona[child.type] = 1

	#collect in the right format
	nextObs = np.array(persona + words + [stage], dtype = np.float)
	return nextObs, interactions, stage

