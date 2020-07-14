import numpy as np


def react2hint1(action, child):
	info = {}
	done = False
	improvement = 0
	penalty = 0.1
	potential = 10 - child.pre_score  # 9 is max score
	quitPenalty = 0

	if action == child.hints[child.neededHint]: #if we gave the correct hint
		child.neededHint += 1
		child.correctHints += 1

	else:
		child.wrongHints += 1

	if child.wrongHints >= 2 and np.random.binomial(1, 0.5) == 1:
		# if we've given all hints that the child needs, we achieve potential minus a penalty for wrong hints
		improvement = child.correctHints / len(child.hints) * potential
		improvement -= penalty * child.wrongHints * potential

		improvement += np.random.normal(0,
		                                 potential / 10)  # add random element to improvement, with std dependent on potential

		quitPenalty = 0.2*potential
		done = True
		info['reaction'] = 'quit'

	elif action >= child.hints[-1]: #if we gave a new hint more or equal to needed, the child does the post-test
		# if we've given all hints that the child needs, we achieve potential minus a penalty for wrong hints
		improvement = child.correctHints / len(child.hints) * potential
		improvement -= penalty * child.wrongHints * potential

		improvement += np.random.normal(0,
		                                potential / 10)  # add random element to improvement, with std dependent on potential


		done = True
		if action == child.hints[-1]:
			info['reaction'] = 'correct'
		else:
			info['reaction'] = 'early'

	if action != 3: #if we did not give the last hint, otherwise it doesn't matter
		child.prev_q[action] = 1

	info['perfect'] = potential
	reward = min(10, max(improvement, 0)) - quitPenalty

	return reward, done, info

def react2hint2(action, child):
	info = {}
	done = False
	improvement = 0
	penalty = 0.5
	potential = 10 - child.pre_score  # 9 is max score
	quitPenalty = 0

	if action == child.hints[child.neededHint]: #if we gave the correct hint
		child.neededHint += 1
		child.correctHints += 1

	else:
		child.wrongHints += 1

	if np.random.binomial(1, 0.01 + 0.1 * child.wrongHints) == 1:
		# if we've given all hints that the child needs, we achieve potential minus a penalty for wrong hints
		improvement = child.correctHints / len(child.hints) * potential
		improvement -= potential * penalty

		improvement += np.random.normal(0,
		                                 potential / 10)  # add random element to improvement, with std dependent on potential
		quitPenalty = 0.2 * potential

		done = True
		info['reaction'] = 'quit'

	elif action == child.hints[-1] and action == child.hints[max(0,child.neededHint-1)]: #if we gave a new hint more or equal to needed, the child does the post-test
		# if we've given all hints that the child needs, we achieve potential minus a penalty for wrong hints
		improvement = child.correctHints / len(child.hints) * potential

		improvement += np.random.normal(0,
		                                potential / 10)  # add random element to improvement, with std dependent on potential


		done = True
		info['reaction'] = 'correct'

	elif action == 3:
		improvement = child.correctHints / len(child.hints) * potential
		improvement -= potential * penalty * 0.5

		improvement += np.random.normal(0,
		                                potential / 10)  # add random element to improvement, with std dependent on potential


		done = True
		info['reaction'] = 'early'

	if action != 3: #if we did not give the last hint, otherwise it doesn't matter
		child.prev_q[action] = 1

	reward = min(10, max(improvement, 0)) - quitPenalty
	info['perfect'] = potential
	return reward, done, info

def react2hint3(action, child):
	info = {}
	done = False
	penalty = 0.2
	potential = 10 - child.pre_score  # 9 is max score
	improvement = 0
	quitPenalty = 0

	if action == child.hints[child.neededHint]: #if we gave the correct hint
		child.neededHint += 1
		child.correctHints += 1

	else:
		child.wrongHints += 1
		if action > child.hints[child.neededHint]:
			child.wrongHintsLarger += 1

	if np.random.binomial(1, 0.05 + 0.15 * child.wrongHints) == 1:
		# if we've given all hints that the child needs, we achieve potential minus a penalty for wrong hints
		improvement = child.correctHints / len(child.hints) * potential
		improvement -= penalty * child.wrongHintsLarger * potential
		improvement -= 0.5 * potential
		quitPenalty = 0.2 * potential
		improvement += np.random.normal(0,
		                                 potential / 10)  # add random element to improvement, with std dependent on potential
		info['reaction'] = 'quit'
		done = True

	elif (action == child.hints[-1] and max(0, child.neededHint-1) == len(child.hints) - 1) or action == 3: #if we gave a new hint more or equal to needed, the child does the post-test
		# if we've given all hints that the child needs, we achieve potential minus a penalty for wrong hints
		improvement = child.correctHints / len(child.hints) * potential
		improvement -= penalty * child.wrongHintsLarger * potential

		improvement += np.random.normal(0,
		                                potential / 10)  # add random element to improvement, with std dependent on potential

		if (action == child.hints[-1] and max(0, child.neededHint-1) == len(child.hints) - 1):
			info['reaction'] = 'correct'
		else:
			info['reaction'] = 'early'

		done = True

	reward = min(10, max(improvement, 0)) - quitPenalty

	info['perfect'] = potential
	return reward, done, info

def react2hint4(action, child):
	done = False
	penalty = 0.5
	potential = 10 - child.pre_score  # 9 is max score
	improvement = 0
	info = {}
	quitPenalty = 0

	if action == child.hints[child.neededHint]: #if we gave the correct hint
		child.neededHint += 1
		child.correctHints += 1

	else:
		child.wrongHints += 1
		if action > child.hints[child.neededHint]:
			child.wrongHintsLarger += 1

	if np.random.binomial(1, min(0.01 + 0.1 * child.wrongHints, 1)) == 1: #if the child quits
		improvement = potential - penalty * potential

		# add random element to improvement, with std dependent on potential
		improvement += np.random.normal(0, potential / 10)
		done = True
		quitPenalty = 0.2 * potential
		info['reaction'] = 'quit'

	# if we gave the correct last hint a child needs
	elif action == child.hints[-1] and action == child.hints[max(0,child.neededHint-1)]:
		improvement = potential

		# add random element to improvement, with std dependent on potential
		improvement += np.random.normal(0, potential / 10)
		done = True
		info['reaction'] = 'correct'

	#if we gave the answer (hint 3) too early
	elif action == 3:
		improvement = potential - 0.7*penalty * potential

		# add random element to improvement, with std dependent on potential
		improvement += np.random.normal(0, potential / 10)
		done = True
		info['reaction'] = 'early'


	if action != 3:
		child.prev_q[action] = 1

	reward = min(10, max(improvement, 0)) - quitPenalty
	info['perfect'] = potential

	return reward, done, info


