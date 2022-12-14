import jsbsim
import gym

import numpy as np

from rendering import Viewer, load_mesh, load_shader, RenderObject, Grid
from quaternion import Quaternion

state_format = [
    "position/lat-gc-rad",
    "position/long-gc-rad",
    "position/h-sl-meters",
    "velocities/mach",
    "aero/alpha-rad",
    "aero/beta-rad",
    "velocities/p-rad_sec",
    "velocities/q-rad_sec",
    "velocities/r-rad_sec",
    "attitude/phi-rad",
    "attitude/theta-rad",
    "attitude/psi-rad",
]

class JSBSimEnv(gym.Env):
    def __init__(self, root='.'):
        self.simulation = jsbsim.FGFDMExec(root, None)
        self.simulation.set_debug_level(0)
        self.simulation.load_model('f16')
        self._set_initial_conditions()
        self.simulation.run_ic()
        self.down_sample = 4
        self.state = np.zeros(12)
        self.goal = np.zeros(3)
        self.viewer = None

    def _set_initial_conditions(self):
        self.simulation.set_property_value('propulsion/set-running', -1)
        self.simulation.set_property_value('ic/u-fps', 900.)
        self.simulation.set_property_value('ic/h-sl-ft', 5000)
    
    def step(self, action):
        roll_cmd, pitch_cmd, yaw_cmd, throttle = action

        self.simulation.set_property_value("fcs/aileron-cmd-norm", roll_cmd)
        self.simulation.set_property_value("fcs/elevator-cmd-norm", pitch_cmd)
        self.simulation.set_property_value("fcs/rudder-cmd-norm", yaw_cmd)
        self.simulation.set_property_value("fcs/throttle-cmd-norm", throttle)

        for _ in range(self.down_sample):
            self.simulation.set_property_value("propulsion/tank/contents-lbs", 1000)
            self.simulation.set_property_value("propulsion/tank[1]/contents-lbs", 1000)
            self.simulation.run()

        self._get_state()

        reward = 0
        done = False
        if self.state[2] < 10:
            reward = -10
            done = True
        if np.sum((self.state[:2] - self.goal[:2])**2) < 1e-6 and abs(self.state[2] - self.goal[2]) < 100:
            reward = 10
            done = True
        
        return np.hstack([self.state, self.goal]), reward, done, {}
    
    def _get_state(self):
        for i, property in enumerate(state_format):
            self.state[i] = self.simulation.get_property_value(property)
    
    def reset(self, seed=None):
        self.simulation.run_ic()
        self.simulation.set_property_value('propulsion/set-running', -1)
        self.simulation.set_property_value("gear/gear-cmd-norm", 0.0)
        self.simulation.set_property_value("gear/gear-pos-norm", 0.0)

        rng = np.random.default_rng(seed)
        distance = rng.random() * .1 + .01
        bearing = rng.random() * 2 * np.pi
        altitude = rng.random() * 3000

        self.goal[:2] = np.cos(bearing), np.sin(bearing)
        self.goal[:2] *= distance
        self.goal[2] = altitude

        self._get_state()

        return self.state.copy()
    
    def render(self):
        scale = 1e-4

        if self.viewer is None:
            self.viewer = Viewer(1280, 720)

            f16_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, "f16.obj")
            self.f16 = RenderObject(f16_mesh)
            self.f16.transform.scale = 1/30
            self.f16.color = 0, 0, .6
            self.viewer.objects.append(self.f16)
            self.viewer.objects.append(Grid(self.viewer.ctx, self.viewer.unlit, 21, 1.))
        
        # Rough conversion from lat/long to meters
        x, y = self.state[:2] * 4e7

        self.f16.transform.z = x * scale
        self.f16.transform.x = -y * scale
        self.f16.transform.y = self.state[2] * scale

        # print(self.f16.transform.position)

        # rot = Quaternion.from_euler(-self.state[10], -self.state[11], self.state[9], mode=1)
        rot = Quaternion.from_euler(*self.state[9:])
        rot = Quaternion(rot.w, -rot.y, -rot.z, rot.x)
        self.f16.transform.rotation = rot

        self.viewer.set_view(-y * scale, self.state[2] * scale + 1, x * scale - 3, Quaternion.from_euler(np.pi/12, 0, 0, mode=1))

        self.viewer.render()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

if __name__ == "__main__":
    from time import sleep
    env = JSBSimEnv()
    env.reset()
    env.render()
    for _ in range(300):
        env.step(np.array([0.05, -0.2, 0, .5]))
        env.render()
        sleep(1/30)
    env.close()