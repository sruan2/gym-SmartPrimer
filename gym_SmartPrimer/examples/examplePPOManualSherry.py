import gym
from gym import wrappers
import numpy as np
from rlgraph.agents import Agent
from rlgraph.environments import OpenAIGymEnv
import json
import os
import pickle
import argparse
import copy
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='example')
parser.add_argument('--seed', type=int, default=0,
                    help='numpy seed ')
parser.add_argument('--time', type=int, default=1,
                    help='numpy seed ')
args = parser.parse_args()
np.random.seed(args.seed)

#create the environment
env = OpenAIGymEnv.from_spec({
				"type": "openai",
				"gym_env": 'gym_SmartPrimer:SmartPrimer-realistic-v2'
		})

#configure the agent settings in this file
agent_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),  'agents/ppoSmartPrimer_config.json')

with open(agent_config_path, 'rt') as fp:
	agent_config = json.load(fp)

#retreive the agent from RLgraph
agent = Agent.from_spec(
				agent_config,
				state_space=env.state_space,
				action_space=env.action_space
		)

#define number of children to simulate
episode_count = 10000

reward = 0
done = False

def evaluate(agent_obs, nChildren):
	envObs = OpenAIGymEnv.from_spec({
		"type": "openai",
		"gym_env": 'gym_SmartPrimer:SmartPrimer-realistic-v2'
	})

	improvements = []
	for i in range(0, nChildren):
		ob_obs = envObs.reset()
		action_list_obs = []

		while True:
			time_percentage_obs = min(agent_obs.timesteps / 1e6, 1.0)
			action = agent_obs.get_action(ob_obs, time_percentage=time_percentage_obs)
			# action = np.random.randint(0, 4)
			# action = 3

			action_list_obs.append(action)
			# print(time_percentage)
			next_ob_obs, reward, done, Baseinfo = envObs.step(action)
			# print(ob)
			# print(reward)
			# print(done)
			# print(action)
			agent_obs.observe(ob_obs, action, None, reward, next_ob_obs, done)
			ob_obs = next_ob_obs

			# DONT FORGET TO DELETE THIS
			ob_obs = [np.random.randint(0, 9), np.random.randint(2, 7), np.random.randint(0, 2), np.random.randint(0, 2),
			      np.random.randint(0, 2), np.random.randint(0, 4), np.random.randint(0, 46), np.random.randint(0, 11)]

			# if agent.timesteps % 200 == 0:
			# 	agent.update(time_percentage=time_percentage)

			# agent.update(time_percentage=time_percentage)

			if done:
				# if (i%50==0):
				# 	print(ob)
				# 	print(reward)
				# 	print(done)

				# print("Student", i, "actions")
				# print(action_list)
				improvements.append(envObs.gym_env.info['improvementPerChild'])

				agent_obs.reset()
				break
	# print('Improvements were: {}'.format(improvements))
	return np.mean(improvements), np.std(improvements), envObs.gym_env.info['actionInfo']


agent_obs = copy.copy(agent)
evaluation_improvement, evaluation_stds, actionInfo = evaluate(agent_obs, 500)
print('encourage: {}'.format(actionInfo['encourage'][-20:]))
print('question: {}'.format(actionInfo['question'][-20:]))
print('hint: {}'.format(actionInfo['hint'][-20:]))
print('nothing: {}'.format(actionInfo['nothing'][-20:]))
a=1


evaluation_improvements = []
for i in range(episode_count):
		#get the new children
		ob = env.reset()
		action_list = []

		while True:
				time_percentage = min(agent.timesteps / 1e6, 1.0)
				action = agent.get_action(ob, time_percentage=time_percentage)
				action_list.append(action)
				#print(time_percentage)
				next_ob, reward, done, Baseinfo = env.step(action)
				# print(ob)
				# print(reward)
				# print(done)
				# print(action)
				agent.observe(ob, action, None, reward, next_ob, done)
				ob = next_ob




				# if agent.timesteps % 200 == 0:
				# 	agent.update(time_percentage=time_percentage)

				# agent.update(time_percentage=time_percentage)
				# print('agent timesteps: {}'.format(agent.timesteps))
				if done:
						# if (i%50==0):
						# 	print(ob)
						# 	print(reward)
						# 	print(done)

						# print("Student", i, "actions")
						# print(action_list)


						if i % 1 ==0:
							agent_obs = copy.copy(agent)
							evaluation_improvement, evaluation_stds, actionInfo = evaluate(agent_obs, 500)
							print('encourage: {}'.format(actionInfo['encourage'][-20:]))
							print('question: {}'.format(actionInfo['question'][-20:]))
							print('hint: {}'.format(actionInfo['hint'][-20:]))
							print('nothing: {}'.format(actionInfo['nothing'][-20:]))
							a=1


						if i % 10 == 0:
							agent.update(time_percentage=time_percentage)

							if i % 50 == 0:
								agent_obs = copy.copy(agent) #copy the policy
								evaluation_improvement, evaluation_stds = evaluate(agent_obs, 500)

								print(evaluation_improvement)
								print(evaluation_stds)
								evaluation_improvements.append(evaluation_improvement)

							# if i % 2000 == 0:
							# 	plt.plot(evaluation_improvements)
							# 	plt.title('Improvement per agent update, averaged over 100 evaluation children')
							# 	plt.xlabel('Number of children trained x20')
							# 	plt.ylabel('Average improvement of 100 evaluation children')
							# 	plt.show()
							# print('Evaluation improvements are: {}'.format(evaluation_improvements))

						agent.reset()
						break

print(env.gym_env.info['Improvement'])

# print(evaluation_improvements)
# plt.plot(evaluation_improvements)
# plt.title('Improvement per agent update, averaged over 100 evaluation children')
# plt.xlabel('Number of children trained x20')
# plt.ylabel('Average improvement of 100 evaluation children')
# plt.show()

# print(env.gym_env.info['Performance'])                       
#make the plots
env.render()



performance = env.gym_env.info['Performance']
improvement = env.gym_env.info['Improvement']
                       
pickle_name = '/Users/williamsteenbergen/Desktop/Smart_Primer/pickles2/per_ppo_psi03_'+str(args.seed)+'.pickle'
with open(pickle_name , 'wb') as handle:
    pickle.dump(performance, handle, protocol=pickle.HIGHEST_PROTOCOL)                       
pickle_name = '/Users/williamsteenbergen/Desktop/Smart_Primer/pickles2/imp_ppo_psi03_'+str(args.seed)+'.pickle'
with open(pickle_name , 'wb') as handle:
    pickle.dump(improvement, handle, protocol=pickle.HIGHEST_PROTOCOL)

actionInfo = env.gym_env.info['actionInfo']    
pickle_name = '/Users/williamsteenbergen/Desktop/Smart_Primer/pickles2/actionInfo_ppo_psi03_'+str(args.seed)+'.pickle'
with open(pickle_name , 'wb') as handle:
    pickle.dump(actionInfo, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    