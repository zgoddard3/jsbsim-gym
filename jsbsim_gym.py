import jsbsim
import gym

import numpy as np

from rendering import Viewer, load_mesh, load_shader, RenderObject, Grid
from quaternion import Quaternion

# Initialize format for the environment state vector
STATE_FORMAT = [
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

STATE_LOW = np.array([
    -np.inf,
    -np.inf,
    0,
    0,
    -np.pi,
    -np.pi,
    -np.inf,
    -np.inf,
    -np.inf,
    -np.pi,
    -np.pi,
    -np.pi,
    -np.inf,
    -np.inf,
    0,
])

STATE_HIGH = np.array([
    np.inf,
    np.inf,
    np.inf,
    np.inf,
    np.pi,
    np.pi,
    np.inf,
    np.inf,
    np.inf,
    np.pi,
    np.pi,
    np.pi,
    np.inf,
    np.inf,
    np.inf,
])

# Radius of the earth
RADIUS = 6.3781e6

class JSBSimEnv(gym.Env):
    """
    ### Description
    Gym environment using JSBSim to simulate an F-16 aerodynamics model with a
    simple point-to-point navigation task. The environment terminates when the
    agent enters a cylinder around the goal or crashes by flying lower than sea
    level. The goal is initialized at a random location in a cylinder around the
    agent's starting position. 

    ### Observation
    The observation is given as the position of the agent, velocity (mach, alpha,
    beta), angular rates, attitude, and position of the goal (concatenated in
    that order). Units are meters and radians. 

    ### Action Space
    Actions are given as normalized body rate commands and throttle command. 
    These are passed into a low-level PID controller built into the JSBSim model
    itself. The rate commands should be normalized between [-1, 1] and the 
    throttle command should be [0, 1].

    ### Rewards
    A positive reward is given for reaching the goal and a negative reward is 
    given for crashing. It is recommended to use the PositionReward wrapper 
    below to eliminate the problem of sparse rewards.
    """
    def __init__(self, root='.'):
        super().__init__()

        # Set observation and action space format
        self.observation_space = gym.spaces.Box(STATE_LOW, STATE_HIGH, (15,))
        self.action_space = gym.spaces.Box(np.array([-1,-1,-1,0]), 1, (4,))

        # Initialize JSBSim
        self.simulation = jsbsim.FGFDMExec(root, None)
        self.simulation.set_debug_level(0)

        # Load F-16 model and set initial conditions
        self.simulation.load_model('f16')
        self._set_initial_conditions()
        self.simulation.run_ic()

        self.down_sample = 4
        self.state = np.zeros(12)
        self.goal = np.zeros(3)
        self.dg = 100
        self.viewer = None

    def _set_initial_conditions(self):
        # Set engines running, forward velocity, and altitude
        self.simulation.set_property_value('propulsion/set-running', -1)
        self.simulation.set_property_value('ic/u-fps', 900.)
        self.simulation.set_property_value('ic/h-sl-ft', 5000)
    
    def step(self, action):
        roll_cmd, pitch_cmd, yaw_cmd, throttle = action

        # Pass control inputs to JSBSim
        self.simulation.set_property_value("fcs/aileron-cmd-norm", roll_cmd)
        self.simulation.set_property_value("fcs/elevator-cmd-norm", pitch_cmd)
        self.simulation.set_property_value("fcs/rudder-cmd-norm", yaw_cmd)
        self.simulation.set_property_value("fcs/throttle-cmd-norm", throttle)

        # We take multiple steps of the simulation per step of the environment
        for _ in range(self.down_sample):
            # Freeze fuel consumption
            self.simulation.set_property_value("propulsion/tank/contents-lbs", 1000)
            self.simulation.set_property_value("propulsion/tank[1]/contents-lbs", 1000)

            # Set gear up
            self.simulation.set_property_value("gear/gear-cmd-norm", 0.0)
            self.simulation.set_property_value("gear/gear-pos-norm", 0.0)

            self.simulation.run()

        # Get the JSBSim state and save to self.state
        self._get_state()

        reward = 0
        done = False

        # Check for collision with ground
        if self.state[2] < 10:
            reward = -10
            done = True

        # Check if reached goal
        if np.sqrt(np.sum((self.state[:2] - self.goal[:2])**2)) < self.dg and abs(self.state[2] - self.goal[2]) < self.dg:
            reward = 10
            done = True
        
        return np.hstack([self.state, self.goal]), reward, done, {}
    
    def _get_state(self):
        # Gather all state properties from JSBSim
        for i, property in enumerate(STATE_FORMAT):
            self.state[i] = self.simulation.get_property_value(property)
        
        # Rough conversion to meters. This should be fine near zero lat/long
        self.state[:2] *= RADIUS
    
    def reset(self, seed=None):
        # Rerun initial conditions in JSBSim
        self.simulation.run_ic()
        self.simulation.set_property_value('propulsion/set-running', -1)
        
        # Generate a new goal
        rng = np.random.default_rng(seed)
        distance = rng.random() * 9000 + 1000
        bearing = rng.random() * 2 * np.pi
        altitude = rng.random() * 3000

        self.goal[:2] = np.cos(bearing), np.sin(bearing)
        self.goal[:2] *= distance
        self.goal[2] = altitude

        # Get state from JSBSim and save to self.state
        self._get_state()

        return np.hstack([self.state, self.goal])
    
    def render(self):
        scale = 1e-3

        if self.viewer is None:
            self.viewer = Viewer(1280, 720)

            f16_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, "f16.obj")
            self.f16 = RenderObject(f16_mesh)
            self.f16.transform.scale = 1/30
            self.f16.color = 0, 0, .4

            goal_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, "cylinder.obj")
            self.cylinder = RenderObject(goal_mesh)
            self.cylinder.transform.scale = scale * 100
            self.cylinder.color = 0, .4, 0

            self.viewer.objects.append(self.f16)
            self.viewer.objects.append(self.cylinder)
            self.viewer.objects.append(Grid(self.viewer.ctx, self.viewer.unlit, 21, 1.))
        
        # Rough conversion from lat/long to meters
        x, y, z = self.state[:3] * scale

        self.f16.transform.z = x 
        self.f16.transform.x = -y
        self.f16.transform.y = z

        rot = Quaternion.from_euler(*self.state[9:])
        rot = Quaternion(rot.w, -rot.y, -rot.z, rot.x)
        self.f16.transform.rotation = rot

        # self.viewer.set_view(-y , z + 1, x - 3, Quaternion.from_euler(np.pi/12, 0, 0, mode=1))

        x, y, z = self.goal * scale

        self.cylinder.transform.z = x
        self.cylinder.transform.x = -y
        self.cylinder.transform.y = z

        r = self.f16.transform.position - self.cylinder.transform.position
        rhat = r/np.linalg.norm(r)
        r += rhat*.5
        x,y,z = r
        angle = np.arctan2(-x,-z)

        self.viewer.set_view(x , y, z, Quaternion.from_euler(np.pi/12, angle, 0, mode=1))

        print(self.cylinder.transform.position)

        # print(self.f16.transform.position)

        # rot = Quaternion.from_euler(-self.state[10], -self.state[11], self.state[9], mode=1)
        

        self.viewer.render()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

class PositionReward(gym.Wrapper):
    """
    This wrapper adds an additional reward to the JSBSimEnv. The agent is 
    rewarded based when movin closer to the goal and penalized when moving away.
    Staying at the same distance will result in no additional reward. The gain 
    may be set to weight the importance of this reward.
    """
    def __init__(self, env, gain):
        super().__init__(env)
        self.gain = gain
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        displacement = obs[-3:] - obs[:3]
        distance = np.linalg.norm(displacement)
        reward += self.gain * (self.last_distance - distance)
        self.last_distance = distance
        return obs, reward, done, info
    
    def reset(self):
        obs = super().reset()
        displacement = obs[-3:] - obs[:3]
        self.last_distance = np.linalg.norm(displacement)
        return obs

# Create entry point to wrapped environment
def wrap_jsbsim(**kwargs):
    return PositionReward(JSBSimEnv(**kwargs))

# Register the wrapped environment
gym.register(
    id="JSBSim-v0",
    entry_point=wrap_jsbsim,
    max_episode_steps=1200
)

# Short example script to create and run the environment with
# constant action for 1 simulation second.
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