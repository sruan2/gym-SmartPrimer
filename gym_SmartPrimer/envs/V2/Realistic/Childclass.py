import numpy as np
import json

class Child:
	"""A child class that contains all the information of a child. Takes info from childConfig.json:
	 Attributes:
	  type: the child type, randomly chosen
	  age: the child's age
	  grade: the child's grade
	  preScore: the child's preScore
	  neededHints: the number of hints a child needs before it can succesfully answer a sub-problem
	  neededTime: the time a child needs to solve a sub-problem (infinite when he/she still needs hints or encouragements)
	  anxiety: the child's anxiety score
	  encouragementsNeeded: the number of encouragements a child needs to succesfully solve a sub-problem

	  nWrongHints: the number of hints a child received after already having received more than neededHints hints
	  nQuestions: the number of questions a child receives
	  nEncouragements: the number of encouragements a child receives

	  """
	def __init__(self, info):

		stochastics = 1

		type = np.random.randint(0, info['nTypes'])

		self.type = type #0 = smart, 1 = outgoing, 2 = quiet, 3 = anxious

		self.age = info['meanAge'][type] + round(np.random.normal(0, 1)) * stochastics

		self.grade = info['meanGrade'][type] + round(np.random.normal(0, 0.5))* stochastics

		self.preScore = min(max(info['meanPreScore'][type] + np.random.randint(-2, 2) * stochastics, 0) , 10)

		self.neededHints = min(max(info['meanNeededHints'][type] + round(np.random.normal(0, 0.5) * stochastics), 0), 4)

		self.anxiety = info['meanAnxietyScore'][type] + round(np.random.normal(0, 2))* stochastics

		self.encouragementsNeeded = max(0, info['meanNeededEncouragements'][type] + round(np.random.normal(0, 0.3)* stochastics))

		if self.neededHints == 0 and self.encouragementsNeeded == 0:
			self.neededTime = max(info['meanNeededTimeAfterHints'][type] + round(np.random.normal(0, 1)* stochastics), 0)
		else:
			self.neededTime = np.inf

		self.nWrongHints = 0
		self.nQuestions = 0
		self.nEncouragements = 0
		self.nHints = 0
		self.psi = 0.3
		self.nWrongAnswers = 0
		self.nCorrectHints = 0
		self.nWrongAnswersCurrent = 0


