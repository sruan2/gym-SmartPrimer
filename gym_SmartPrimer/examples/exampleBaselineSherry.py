import gym
import numpy as np
import gym_SmartPrimer.agents.baselineV2 as Baseline
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
        # if (i%500==0):
        #     print(ob)
        #     print(reward)
        #     print(done)
        if done:
            agent.reset()
            break

    
#make the plots
env.render()
performance = env.info['Performance']
improvement = env.info['Improvement']
                       
pickle_name = '/Users/jiequanzhang/Desktop/smart_primer/gym-SmartPrimer/pickles/per_baseline_'+str(args.seed)+'.pickle'
with open(pickle_name , 'wb') as handle:
    pickle.dump(performance, handle, protocol=pickle.HIGHEST_PROTOCOL)                       
pickle_name = '/Users/jiequanzhang/Desktop/smart_primer/gym-SmartPrimer/pickles/imp_baseline_'+str(args.seed)+'.pickle'
with open(pickle_name , 'wb') as handle:
    pickle.dump(improvement, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
actionInfo = env.info['actionInfo']    
pickle_name = '/Users/jiequanzhang/Desktop/smart_primer/gym-SmartPrimer/pickles/actionInfo_baseline_'+str(args.seed)+'.pickle'
with open(pickle_name , 'wb') as handle:
    pickle.dump(actionInfo, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    