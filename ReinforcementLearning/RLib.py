# Import the RL algorithm (Algorithm) we would like to use.
import os

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.a2c import A2C
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.dqn import DQNConfig
import Environment
from torchvision import models

# Configure the algorithm.
config = {
    "framework": "torch",
    "double_q": True,
    "dueling": False,
    "num_atoms": 1,
    "noisy": False,
    "replay_buffer_config": {
        "type": "MultiAgentReplayBuffer",
        "capacity": 1000000
    },
    "num_steps_sampled_before_learning_starts": 20000,
    "n_step": 1,
    "target_network_update_freq": 8000,
    "lr": 0.0000625,
    "adam_epsilon": 0.00015,
    "hiddens": [
        512
    ],
    "rollout_fragment_length": 4,
    "train_batch_size": 32,
    "exploration_config": {
        "epsilon_timesteps": 200000,
        "final_epsilon": 0.01
    },
    "num_gpus": 1,
    "min_sample_timesteps_per_iteration": 50000,

    "num_workers": 2,
}

# Create our RLlib Trainer.
ray.rllib.utils.check_env(Environment.SimpleACC(config))
algo = DQN(env=Environment.SimpleACC, config=config)

policy = algo.get_policy()
print("policy:")
# policy.export_model(os.getcwd(),15)
print(policy.model)
algo.export_policy_model(os.getcwd())

# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.
for _ in range(5):
    print(pretty_print(algo.train()))

# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
# algo.evaluate()


print('Training done')
print('Simulating:')
env = Environment.SimpleACC(config)

# run until episode ends
episode_reward = 0
done = False
obs = env.reset()
while not done:
    action = algo.compute_single_action(observation=obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
env.plot()
