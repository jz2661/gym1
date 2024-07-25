
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import LEFT, RIGHT, DOWN, UP
from gymnasium import spaces, Env
import numpy as np

class MySim(Env):
    def __init__(self):
        #low = np.array([-1, -1], dtype=np.float32)
        #high = np.array([1,1], dtype=np.float32)

        self.action_space = spaces.Box(low=0.0, high=1.0, dtype=np.float32)
        
        self.observation_space = spaces.MultiDiscrete([100, 50])

        self.rooms = 100 # remaining rooms
        self.days = 50
        self.rewards = 0

        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([self.rooms, self.days]), dtype=np.int32)

    def _demand(self, action):
        # given price, return booking
        return min(10*(1-action[0]), self.rooms)

    def _get_obs(self):
        #return {"rooms": self.rooms, "days": self.days}
        return np.array([self.rooms, self.days], dtype=np.int32)

    def _get_info(self):
        return {"rewards": self.rewards}

    def step(self, action):
        self.days -= 1
        bookings = self._demand(action)
        
        self.rooms -= bookings

        observation = self._get_obs()
        reward = float(bookings * action)
        self.rewards += reward

        terminated = (self.days == 0)
        
        info = self._get_info()

        return observation, reward, terminated, False, info

    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.rooms = 100 # remaining rooms
        self.days = 50
        self.rewards = 0
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def render(self,mode="human"):
        pass

    def seed():
        pass