import gym
import jsbsim_gym
import imageio as iio
from os import path
from features import JSBSimFeatureExtractor
from stable_baselines3 import SAC

policy_kwargs = dict(
    features_extractor_class=JSBSimFeatureExtractor
)

env = gym.make("JSBSim-v0")

model = SAC.load("models/jsbsim_sac", env)

mp4_writer = iio.get_writer("video.mp4", format="ffmpeg", fps=30)
gif_writer = iio.get_writer("video.gif", format="gif", fps=5)
obs = env.reset()
done = False
step = 0
while not done:
    render_data = env.render(mode='rgb_array')
    mp4_writer.append_data(render_data)
    if step % 6 == 0:
        gif_writer.append_data(render_data)

    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = env.step(action)
    step += 1
mp4_writer.close()
gif_writer.close()
env.close()