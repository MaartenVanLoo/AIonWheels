# AIonWheels
I-DistributedAI Reinforcement Learning

# DQN
Qlearner class consists of the Deep Q-Learning model, and is used to train our agent. It can load in previous models, or save the model of trained agents to reuse them later.
It also chooses actions given a certain state. It also contains a Replay Buffer.
Commands DQN:
- forward: forwards data through the neural network
- act: chooses the action that the agent takes=> epsilon greedy and discrete action space
Commands QLearning:
- save: to save the model
- load: to load a model
- __epsilon_by_frame: creates depleting epsilon, depending on current frame
- __update: updates the qvalues and loss
- train: to train the agent
- eval: when in evaluation mode, to evaluate the loaded model

---
#Simple environment
The Environment class is the simple environment to train the agents for the first time. It has different kind of leading cars, so the agent can train on different behaviours/situations.
It also contains a dictionary "config" to set different parameters of the simple environment.

To train in the simple environment simply run the environment file.
```cmd
python Environment.py
```
The model can be configured using the config in Environment.py
```python
config = {
    'device': 'cuda',       # select the device (cuda or cpu)
    'batch_size': 2048,     # batch size used for update
    'mini_batch': 24,       # only update once after n experiences
    'num_frames': 2000000,  # total number of frames used to train
    'gamma': 0.90,          # discount factor
    'replay_size': 250000,  # size of replay buffer
    'lr': 0.0003,           # learning rate
    'reward_offset': 1.5,   # reward offset
    
    # model parameters:
    'history_frames': 3,    # number of states in the history
    'num_inputs': 6,        # =size of states!    
    'num_actions': 11,      # number of actions used to discretise action space
    'hidden': [128, 512, 512, 128, 64], #hidden layer sizes
    'debug': False,
}
```

---
#Carla environment
After training the agent in the simple environment, we can use the saved model to train further in the CarlaEnvironment class.
This class connects to the Carla environment and makes it possible to train the model of our agent. After training in this class, we have our final cruise control agent.
Also the CarlaEnvironment has a config dictionary to set the configurations.
It is important to set this config for the loaded model (history frames, inputs, actions, hidden)\
Commands:
- reset: reset the environment to start new episode
- step: takes a timestep=> calculates new state and reward
- eval/train: to set evaluation mode or training mode
- __getState: returns state
- __getReward: calculates reward, given the taken action, distance and vehicle in front
- __isTerminal: checks for collisions or terminal states
- __getSafeDistance: calculates the safe distance to keep, depending on speed
- __getTargetSpeed: returns the speed that the agent should drive, depending on car in front

To train in Carla you need to have a carla instance running either locally or on a different host. The host can be set using the --host option, by default set to 127.0.0.1.
```cmd
python --host [ip] CarlaEnvironment/CarlaRLEnvFast.py
```
The model can be configured using the config in CarlaRLEnvFast.py
```python
config = {
    'device': 'cuda',       # select the device (cuda or cpu)
    'batch_size': 2048,     # batch size used for update
    'mini_batch': 4,        # only update once after n experiences
    'num_frames': 100000,   # total number of frames used to train
    'gamma': 0.90,          # discount factor
    'replay_size': 250000,  # discount factor
    'lr': 0.0003,           # learning rate
    'reward_offset': 2.5,   # reward offset
    #epsilon greedy parameters
    'epsilon_start':0.5,    # initial factor
    'epsilon_final': 0.04,  # final epsilon value
    'epsilon_decay':9000,   # speed of decay
    
    'target_update_freq':2000, # update target after n frames
    # model parameters (must be the same as the simple environments in case of a loaded model:
    'history_frames': 3,        # number of states in the history  
    'num_inputs': 12,           # =size of states!
    'num_actions': 101,         # number of actions used to discretize action space
    'hidden': [128, 512, 512, 128, 64], # hidden layer sizes
    'debug':False,
}
```