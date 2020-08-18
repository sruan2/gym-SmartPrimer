import numpy as np

class BaselineAgent(object):
    """Baseline agent that chooses random actions"""
    def __init__(self, action_space):
        self.action_space = action_space
        print(action_space)

    def act(self, observation, reward, done):
        return np.random.randint(0, 4)

    def act2(self, observation, reward, done):
        return 3

    def reset(self):
        pass
