# AIonWheels
I-DistributedAI Reinforcement Learning

Qlearner class consists of the Deep Q-Learning model, and is used to train our agent. It can load in previous models, or save the model of trained agents to reuse them later.
It also chooses actions given a certain state. It also contains a Replay Buffer.\
Commands DQN:\
forward: forwards data through the neural network\
act: chooses the action that the agent takes=> epsilon greedy and discrete action space\
Commands QLearning:\
save: to save the model\
load: to load a model\
__epsilon_by_frame: creates depleting epsilon, depending on current frame\
__update: updates the qvalues and loss\
train: to train the agent\
eval: when in evaluation mode, to evaluate the loaded model\



The Environment class is the simple environment to train the agents for the first time. It has different kind of leading cars, so the agent can train on different behaviours/situations.
It also contains a dictionary "config" to set different parameters of the simple environment.

After training the agent in the simple environment, we can use the saved model to train further in the CarlaEnvironment class.
This class connects to the Carla environment and makes it possible to train the model of our agent. After training in this class, we have our final cruise control agent.
Also the CarlaEnvironment has a config dictionary to set the configurations.
It is important to set this config for the loaded model (history frames, inputs, actions, hidden)\
Commands:\
reset: reset the environment to start new episode\
step: takes a timestep=> calculates new state and reward\
eval/train: to set evaluation mode or training mode\
__getState: returns state\
__getReward: calculates reward, given the taken action, distance and vehicle in front\
__isTerminal: checks for collisions or terminal states\
__getSafeDistance: calculates the safe distance to keep, depending on speed\
__getTargetSpeed: returns the speed that the agent should drive, depending on car in front\
