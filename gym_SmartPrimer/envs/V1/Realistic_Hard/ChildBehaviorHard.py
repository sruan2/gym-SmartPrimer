import numpy as np


def react2hint(action, child):
	done = False

	reward = 0
	penalty = 1
	potential = 9 - child.pre_score  # 9 is max score


	if action == child.hints[child.neededHint]: #if we gave the correct hint
		child.neededHint += 1
		child.correctHints += 1

	else:
		child.wrongHints += 1

	if child.wrongHints >= 2 and np.random.binomial(1, 0.5) == 1:
		# if we've given all hints that the child needs, we achieve potential minus a penalty for wrong hints
		improvement = child.correctHints / len(child.hints) * potential
		improvement -= penalty * child.wrongHints

		improvement += np.random.normal(0,
		                                 potential / 10)  # add random element to improvement, with std dependent on potential

		reward = -2
		done = True

	elif action >= child.hints[-1]: #if we gave a new hint more or equal to needed, the child does the post-test
		# if we've given all hints that the child needs, we achieve potential minus a penalty for wrong hints
		improvement = child.correctHints / len(child.hints) * potential
		improvement -= penalty * child.wrongHints

		improvement += np.random.normal(0,
		                                potential / 10)  # add random element to improvement, with std dependent on potential

		reward = improvement
		done = True

	if action != 3: #if we did not give the last hint, otherwise it doesn't matter
		child.prev_q[action] = 1

	return reward, done

def nextObservation(observation_space, child):
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
