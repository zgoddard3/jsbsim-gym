import gym
import jsbsim_gym
from os import path
from features import JSBSimFeatureExtractor
from stable_baselines3 import SAC

policy_kwargs = dict(
    features_extractor_class=JSBSimFeatureExtractor
)

env = gym.make("JSBSim-v0")

log_path = path.join(path.abspath(path.dirname(__file__)), 'logs')

try:
    model = SAC('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=log_path, gradient_steps=-1)
    model.learn(1000000)
finally:
    model.save("models/jsbsim_sac")
    model.save_replay_buffer("models/jsbsim_sac_buffer")