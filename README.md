# Sherry's notes
To run

`python -m gym_SmartPrimer.examples.exampleBaseline`

`python -m gym_SmartPrimer.examples.examplePPO`

# Smartprimer simulation environment

This repository contains an OpenAI gym environment for simulating high school children that try to solve a problem and
learn a task. The goal of an algorithm in this simulation is to generate hints for the child that correspond to the
hints the child needs. The children will react to hints with words, or will finish or quit the exercise. Before the
children start the exercise, the will make a pre-test. After the children are finished, they make a post-test. The
performance of the algorithm is determined by the average difference between the post-test and pre-test.


The simulation is done by simulating the following tasks
* Doing a pre-test that is scored from 0 to 100
* Use words that correspond to the hint that the child needs
* React to hints that are given to the child by:
    * Quiting
    * Using new words
    * Finish the assignment
* Do the post-test

There are four different environments in V1, and one in V2. It is recommended to work with the V2.


One can run any of these environments by cloning this repo and then loading the environment with:

#### For V1:

`gym.make('gym_SmartPrimer:SmartPrimer-deterministic-v1')` (for deterministic)

`gym.make('gym_SmartPrimer:SmartPrimer-medium-v1')` (for medium)

`gym.make('gym_SmartPrimer:SmartPrimer-realistic-v1')` (for realistic/hard)

`gym.make('gym_SmartPrimer:SmartPrimer-dynamic-v1')` (for realistic/dynamic children)

#### For V2:

`gym.make('gym_SmartPrimer:SmartPrimer-realistic-v2')` (for realistic/dynamic children)

See the documentation for V1 [here](gym_SmartPrimer/envs/V1/README.md) and for V2 [here](gym_SmartPrimer/envs/V2/README.md).



