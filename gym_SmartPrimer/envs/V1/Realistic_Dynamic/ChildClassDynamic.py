import numpy as np

class Child:
	def __init__(self, type, hints):
		self.type = type #type is from not smart -> smart
		self.hints = hints[type]

		#age = [9 + type * (11-9)/nHints + np.random.normal(means['Age'], stds['Age'])] #ranges from around 9 to 11 +- N(mean,std)
		#grade = [round(3 + type * (6-3)/nHints + np.random.normal(means['Grade'], stds['Grade']))] #ranges from around grade 3 to 6 +- N(means, grade)

		scores = [0,1,1,2,3,4,4,5,6,7,7,8,9,9,9]
		self.pre_score = max(min(scores[type] + round(np.random.normal(0,1.5)),9) ,0)

		#init words 3 type of words can be used
		prev_hints = [0,0,0] #initializing 4 previous hints

		self.neededHint = 0
		self.prev_q = prev_hints
		self.wrongHints = 0
		self.wrongHintsLarger = 0
		self.correctHints = 0