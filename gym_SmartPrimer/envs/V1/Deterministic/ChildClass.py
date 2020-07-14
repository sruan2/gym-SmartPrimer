import numpy as np

class Child:
	def __init__(self, type, hints):
		self.type = type #type is from not smart -> smart
		self.hints = hints[type]

		#age = [9 + type * (11-9)/nHints + np.random.normal(means['Age'], stds['Age'])] #ranges from around 9 to 11 +- N(mean,std)
		#grade = [round(3 + type * (6-3)/nHints + np.random.normal(means['Grade'], stds['Grade']))] #ranges from around grade 3 to 6 +- N(means, grade)
		if self.type <= 7:
			pre_score = 0
		elif self.type <= 11:
			pre_score = 1
		else:
			pre_score = 2

		#init words 3 type of words can be used
		words = [0,0,0,0]  # initializing
		prev_hints = [0,0,0] #initializing 4 previous hints

		self.pre_score = pre_score
		self.words = words
		self.neededHint = 0
		self.wrongHints = 0
		self.prev_q = prev_hints