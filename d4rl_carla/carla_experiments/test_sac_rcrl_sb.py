import numpy as np

from rcrl import RCRLSAC
# from stable_baselines3.sac import CnnPolicy
from rcrl_policy import RCRLPolicy
import gym
import d4rl
import json
import os

env = gym.make('carla-lane-render-rcrl-v0')
exp_name = "RCRL_carla"

model = RCRLSAC.load(exp_name)

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
