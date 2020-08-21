import gym
import numpy as np
import gym_SmartPrimer.agents.baselineV2 as Baseline

np.random.seed(2)

#create the environment
env = gym.make('gym_SmartPrimer:SmartPrimer-realistic-v2')
agent = Baseline.BaselineAgent(env.action_space)

#define number of children to simulate
episode_count = 500

reward = 0
done = False

for i in range(episode_count):
    #get the new children
    ob = env.reset()
    while True:
        action = agent.act(ob, reward, done)
        ob, reward, done, Baseinfo = env.step(action)
        # if (i%100==0):
        #     print(ob)
        #     print(reward)
        #     print(done)
        if done:
            agent.reset()
            break

#make the plots
env.render()

