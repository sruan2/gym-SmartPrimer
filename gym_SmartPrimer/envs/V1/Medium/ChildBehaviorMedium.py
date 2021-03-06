import numpy as np

def react2hint(action, child):
	done = False

	if action == child.hints[child.neededHint]: #if we gave the correct hint
		child.neededHint += 1

		if np.random.binomial(1, 0.9) == 1:
			reward = 1
		else:
			reward = -0.5
	else:
		child.wrongHints += 1
		if np.random.binomial(1, 0.9) == 1:
			reward = -0.5
		else:
			reward = 1

	if child.wrongHints >=2 and np.random.binomial(1, 0.9) == 1:
		reward = -2
		done = True

	elif action >= child.hints[-1]: #if we gave a new hint more or equal to needed
		done = True

	if action != 3: #if we did not give the last hint, otherwise it doesn't matter
		child.prev_q[action] = 1

	return reward, done

def nextObservation(observation_space, child):
	#the new words
	newWordW2V = [0, 0, 0, 0]

	for i in range(0, len(newWordW2V)):
		if i == child.hints[child.neededHint]:
			newWordW2V[i] = min(2, np.random.normal(2,1))
		else:
			newWordW2V[i] = min(2, max(-2, np.random.normal(-2, 1)))

	#The new previous questions
	prev_q = child.prev_q

	# float since RLgraph requires that
	nextObs = np.array([child.pre_score] + newWordW2V + prev_q, dtype=np.float)
	return nextObs
