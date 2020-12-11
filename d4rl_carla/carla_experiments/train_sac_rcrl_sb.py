import numpy as np

from rcrl import RCRLSAC
# from stable_baselines3.sac import CnnPolicy
from rcrl_policy import RCRLPolicy, CnnPolicy
import gym
import d4rl
import json
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

# env = gym.make("carla-lane-render-rcrl-v0")
env = gym.make("carla-lane-v0")
env = Monitor(env)
exp_name = "RCRL_carla"
total_timesteps = 1000000
save_every = 10000
prior_dim = 2
num_steps = 1000

tensorboard_log = os.path.join("./logs", exp_name)

n_actions = env.action_space.shape[0]
print("n actions: ", n_actions)
noise_std = 0.3
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))

model = RCRLSAC(RCRLPolicy, env, action_noise=action_noise, verbose=1, buffer_size=10000, tensorboard_log=tensorboard_log) # train_freq=num_steps, gradient_steps=num_steps,
model.init_replay_buffer()

reward_log = {}
for i in range(total_timesteps // save_every):
    model.learn(total_timesteps=save_every, log_interval=4, reset_num_timesteps=False)
    # done = False
    # total_reward = []
    # obs = env.reset()
    # for i in range(3):
    #     i_reward = 0
    #     steps = 0
    #     while not done and steps < num_steps:
    #         action, _states = model.predict(obs, deterministic=True)
    #         # Perform action
    #         obs, reward, done, _ = env.step(action)
    #         print(steps, action)
    #         i_reward += reward 
    #         steps += 1
    #     total_reward.append(i_reward)
    
    # reward_log[i] = (np.mean(total_reward), np.std(total_reward))
    # with open(os.path.join(tensorboard_log, "reward_log.json"), "w") as f:
    #     json.dump(reward_log, f)
    obs = env.reset()
    
    model.save(exp_name)

model.save(exp_name)

# del model # remove to demonstrate saving and loading

# model = SAC.load(exp_name)

# obs = env.reset()

# while True:
#     done = False
#     while not done:
#         action, _states = model.predict(obs, deterministic=True)
#         # Perform action
#         obs, reward, done, _ = env.step(action)
#         print(action, reward)
#         env.render()
    
#     obs = env.reset()
