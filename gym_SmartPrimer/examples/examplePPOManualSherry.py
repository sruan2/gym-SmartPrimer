import gym
from gym import wrappers
import numpy as np
from rlgraph.agents import Agent
from rlgraph.environments import OpenAIGymEnv
import json
import os
import pickle
import argparse





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
episode_count = 500

reward = 0
done = False

for i in range(episode_count):
		#get the new children
		ob = env.reset()

		while True:
				time_percentage = min(agent.timesteps / 1e6, 1.0)
				action = agent.get_action(ob, time_percentage=time_percentage)
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

				if done:
						# if (i%50==0):
						# 	print(ob)
						# 	print(reward)
						# 	print(done)
						if i % 10 == 0:
							agent.update(time_percentage=time_percentage)
						agent.reset()
						break

# print(env.gym_env.info['Performance'])                       
#make the plots
env.render()

performance = env.gym_env.info['Performance']
improvement = env.gym_env.info['Improvement']
                       
# pickle_name = '/Users/jiequanzhang/Desktop/smart_primer/gym-SmartPrimer/pickles/per_ppo_pen_'+str(args.seed)+'.pickle'
# with open(pickle_name , 'wb') as handle:
#     pickle.dump(performance, handle, protocol=pickle.HIGHEST_PROTOCOL)                       
# pickle_name = '/Users/jiequanzhang/Desktop/smart_primer/gym-SmartPrimer/pickles/imp_ppo_pen_'+str(args.seed)+'.pickle'
# with open(pickle_name , 'wb') as handle:
#     pickle.dump(improvement, handle, protocol=pickle.HIGHEST_PROTOCOL)

