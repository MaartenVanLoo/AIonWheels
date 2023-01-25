# AIonWheels
I-DistributedAI Reinforcement Learning

Qlearner class consists of the Deep Q-Learning model, and is used to train our agent. It can load in previous models, or save the model of trained agents to reuse them later.
It also chooses actions given a certain state. It also contains a Replay Buffer.

The Environment class is the simple environment to train the agents for the first time. It has different kind of leading cars, so the agent can train on different behaviours/situations.

After training the agent in the simple environment, we can use the saved model to train further in the CarlaEnvironment class.
This class connects to the Carla environment and makes it possible to train the model of our agent. After training in this class, we have our final cruise control agent.