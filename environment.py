import gym
from gym import spaces
import numpy as np

class ResourceAllocationEnv(gym.Env):
    """ResourceAllocationEnv that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, n_resources=4, n_steps=10):
        super(ResourceAllocationEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(n_resources,), dtype=np.float32)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=-1000.0, high=1000.0, shape=(n_resources,), dtype=np.float32)
        self.n_resources = n_resources
        self.n_steps = n_steps

        self.state = np.array([0.0 for _ in range(n_resources)])
        self.t = 0
        

    def step(self, action):
        # Execute one time step within the environment
        old_state_value = self.state[0]

        self.state += np.array(action)
        self.state -= np.mean(self.state)

        reward = self.state[0] - old_state_value

        self.t += 1
        done = self.t >= self.n_steps

        info = {"t": self.t}
        
        return self.state, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = np.array([0.0 for _ in range(self.n_resources)])
        self.t = 0
        return self.state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

if __name__ == "__main__":
    env = ResourceAllocationEnv()

    obs = env.reset()
    while True:
        # action = env.action_space.sample()
        action = np.array([10.0, -10.0, -10.0, -10.0])
        obs, reward, done, info = env.step(action)
        print(f"Step: {info['t']}, Action: {action - np.mean(action)}, Observation: {obs}, Reward: {reward}, Done: {done}")
        if done:
            break
    