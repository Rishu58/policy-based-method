# Navigation
This project is part of the Deep Reinforcement Learning Nanodegree Program, by Udacity.



# Understanding the environment
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.



# State description

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

To solve the environment, the Agent must obtain an average score of +30 over 100 consecutive episodes.



# Learning algorithm
For this project, the Deep Deterministic Policy Gradient (DDPG) algorithm was used to train the agent.

Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

This approach is closely connected to Q-learning, and is motivated the same way.

Some characteristics of DDPG:

DDPG is an off-policy algorithm.
DDPG can only be used for environments with continuous action spaces.
DDPG can be thought of as being deep Q-learning for continuous action spaces.


# Running the code

To successfully run the code, you need to ensure that the following files are in place:

Download the depandicies from requirement.txt file.

1. **Navigation.ipynb** - This is the primary notebook where the final implementation is done. Start running the code here.
2. **Model_1.py** - This file contains pytorch Model.
3. **DDPG.py** - This has all the implementation object regarding the algorithm.

Here is the final result which you can expect result.



# Model Architecture


The model architectures for the two neural networks used for the Actor and Critic are as follows:

<img width="584" alt="Screenshot 2024-12-29 at 5 28 35 PM" src="https://github.com/user-attachments/assets/95798a71-76d4-4025-b630-4ad441a31679" />

Actor:

Fully connected layer 1: Input 33 (state space), Output 128, RELU activation, Batch Normalization
Fully connected layer 2: Input 128, Output 128, RELU activation
Fully connected layer 3: Input 64, Output 4 (action space), TANH activation
Critic:

Fully connected layer 1: Input 33 (state space), Output 128, RELU activation, Batch Normalization
Fully connected layer 2: Input 128, Output 128, RELU activation
Fully connected layer 3: Input 64, Output 1

# Hyperparameter

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.95            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 2e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay








