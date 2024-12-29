# Navigation
This project is part of the Deep Reinforcement Learning Nanodegree Program, by Udacity.
The goal in this project is to create and train an agent to navigate and collect bananas in a large, square world.


# Understanding the environment
This environment has been built using the Unity Machine Learning Agents Toolkit (ML-Agents), which is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. You can read more about ML-Agents by perusing the GitHub repository.

The project environment provided by Udacity is similar to, but not identical to the Banana Collector environment on the Unity ML-Agents GitHub page.

In this environment, an Agent navigates a large, square world collecting bananas. Each episode of this task is limited to 300 steps. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible, avoiding the blue ones.

# State description
The state-space has 37 dimensions and contains the agent's velocity, along with the ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward
1 - move backward
2 - turn left
3 - turn right
Solving the environment
To solve the environment, the Agent must obtain an average score of +13 over 100 consecutive episodes.

Getting the code
You have two options to get the code contained in this repository:

Option 1. Download it as a zip file
Click here to download all the content of this repository as a zip file
Uncompress the downloaded file into a folder of your choice
Option 2. Clone this repository using Git version control system
If you are not sure about having Git installed in your system, run the following command to verify that:

$ git --version
If you need to install it, follow this link to do so.

Having Git installed in your system, you can clone this repository by running the following command:

$ git clone https://github.com/silviomori/udacity-deep-reinforcement-learning-p1-navigation.git


# Running the code

To successfully run the code, you need to ensure that the following files are in place:

Download the depandicies from requirement.txt file.

1. **Navigation.ipynb** - This is the primary notebook where the final implementation is done. Start running the code here.
2. **AgentDQ.py** - This file contains all the necessary methods required for the code to function properly. Ensure it is imported into the notebook.

Here is the final result which you can expect result.

<img width="358" alt="image" src="https://github.com/user-attachments/assets/9449920a-7f29-4af8-8c23-6e9baba30934">


