[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Deep Q-network for Unity Banana Collection

## Introduction to the Environment

Unity collect bananas environment in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started

### 1. Install Unity Banana Collection environment 
    Download the environment from one of the links below.  You need only select the environment that matches your operating system:
   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
   - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
   (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

   (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

   Place the file in the GitHub repository, in the `DQN-Navigation/` folder, and unzip (or decompress) the file. 

### 2. Create (and activate) a new environment with Python 3.6.
   - __Linux__ or __Mac__: 
	```bash
	conda create --name dqn python=3.6
	source activate drlnd
	```
	
   - __Windows__: 
	```bash
	conda create --name dqn python=3.6 
	activate dqn
	```

### 3. Install dependencies 
    conda install numpy torch collections matplotlib

## Instructions

### Flow
   - __Train__: The program will do training by default. If training is not needed go with option "--no-training"
   - __Test__:  The program will NOT do testing by default. If testing is required, use option "--test"
   
### The file to store neural weight 
   The program use "checkpoint.pth" to store neural weight by default. The filename could be changed by "-nfile [your file name]"

### Learning algorithms
   Two algorithms are implemented: original DQN and double DQN. The program will run original DQN by default. To switch to double DQN, use "--ddqn"
   
### Examples
   - __A__ : python Navigation.py --test --no-train -nfile 'myneural.pth'
   - __B__ : python Navigation.py --test --ddqn
 

## TODO
We plan to include the feature of Prioritized Experience Replay. It has not been finished yet.
 
