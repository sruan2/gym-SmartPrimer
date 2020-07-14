import numpy as np

def nextObservation1(child):
	#the new words
	newWordW2V = [-2, -2, -2, -2]

	for i in range(0, len(newWordW2V)):
		if i == child.hints[child.neededHint]:
			newWordW2V[i] = min(2, np.random.normal(1,1))
		else:
			newWordW2V[i] = min(2, max(-2, np.random.normal(-1, 1)))


	#The new previous questions
	prev_q = child.prev_q

	# float since RLgraph requires that
	nextObs = np.array([child.pre_score] + newWordW2V + prev_q, dtype=np.float)
	return nextObs

def nextObservation2(child, prev_action, prevNeededHint):
	#the new words
	newWordW2V = list(np.append(np.random.normal([0, 0, 0.5], 1), [0]))

	if prev_action == -1:
		if child.hints[prevNeededHint] == 0:
			newWordW2V[2] = np.random.normal(2, 1)
		else:
			newWordW2V[1] = np.random.normal(1, 1)
			newWordW2V[2] = np.random.normal(1, 1)
	else:
		if prev_action < child.hints[prevNeededHint]:
			newWordW2V[0] = np.random.normal(1, 1)

		elif prev_action == child.hints[prevNeededHint]:
			newWordW2V[1] = np.random.normal(1,1)
			newWordW2V[2] = np.random.normal(1,1)
		elif prev_action > child.hints[prevNeededHint]:
			newWordW2V[2] = np.random.normal(2,1)

	#The new previous questions
	prev_q = child.prev_q

	# float since RLgraph requires that
	nextObs = np.array([child.pre_score] + newWordW2V + prev_q, dtype=np.float)
	return nextObs