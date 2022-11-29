# Import the RL algorithm (Algorithm) we would like to use.
import os

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.a2c import A2C
import ray.rllib.algorithms.apex_dqn.apex_dqn as apex_dqn
from ray.rllib.algorithms.apex_dqn import ApexDQN
from ray.rllib.algorithms.apex_dqn import ApexDQNConfig

from ray.tune.logger import pretty_print
from ray.rllib.algorithms.dqn import DQNConfig
import Environment
from torchvision import models

# Configure the algorithm.
"""
config = {
    "gamma": 0.99,
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
}"""

config = (
            apex_dqn.ApexDQNConfig()
            .framework("torch")
            .rollouts(num_rollout_workers=3,
                      rollout_fragment_length=8)
            .resources(
                num_gpus=0.6,
                num_gpus_per_worker=0.1)
            .training(
                gamma=0.99,
                num_steps_sampled_before_learning_starts=20000,
                optimizer={
                    "num_replay_buffer_shards": 1,

                },
                replay_buffer_config= {
                    "no_local_replay_buffer": True,
                    "type": "MultiAgentPrioritizedReplayBuffer",
                    "capacity": 50000,
                    "prioritized_replay_alpha": 0.6,
                    # Beta parameter for sampling from prioritized replay buffer.
                    "prioritized_replay_beta": 0.4,
                    # Epsilon to add to the TD errors when updating priorities.
                    "prioritized_replay_eps": 1e-6,
                },

                train_batch_size= 64,

            )
            .exploration(
                exploration_config={
                    'epsilon_timesteps' : 200000,
                    'final_epsilon' : 0.01
                }
            )
            .reporting(
                min_sample_timesteps_per_iteration=100000,
                min_time_s_per_iteration=1,
            )
        ).to_dict()
# Create our RLlib Trainer.
ray.rllib.utils.check_env(Environment.SimpleACC(config))
algo = ApexDQN(env=Environment.SimpleACC, config=config)

policy = algo.get_policy()
print("policy:")
# policy.export_model(os.getcwd(),15)
print(policy.model)
algo.export_policy_model(os.getcwd())


env = Environment.SimpleACC(config)
# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.
for _ in range(5):
    print(pretty_print(algo.train()))

    # run until episode ends
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = algo.compute_single_action(observation=obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
    env.plot()
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
