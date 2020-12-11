import numpy as np

from stable_baselines3 import SAC
# from stable_baselines3.sac import CnnPolicy
from stable_baselines3.sac import MlpPolicy
import gym
import d4rl
import json
import os

env = gym.make("carla-lane-render-v0")
exp_name = "baseline_carla"

model = SAC.load(exp_name)

obs = env.reset()

while True:
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        # Perform action
        obs, reward, done, _ = env.step(action)
        print(action, reward)
        env.render()
    
    obs = env.reset()
