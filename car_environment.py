import numpy as np
import gym, math

np.random.seed(0)
"""
This file contains the definition of the environment
in which the agents are run.
"""

gym.logger.set_level(40)


class Mountain:
    possible_actions = [0, 1, 2]

    def __init__(self):
        """Instanciate a new environement in its initial state."""
        self.env = gym.make("MountainCar-v0")
        self.state = self.env.reset()[0]
        # self.env.seed(np.random.randint(1, 10000000))
        self.nb_step = 0

    def sample_action(self):
        return self.env.action_space.sample()

    def reset(self):
        self.nb_step = 0
        self.state = self.env.reset()[0]

    def render(self):
        self.env.render()

    def observe(self):
        """Returns the current observation that the agent can make
        of the environment, if applicable.
        """
        return self.state

    def step_from_state(self, state, action):
        """Simulate act method from a given state"""
        assert self.env.observation_space.contains(state), f"{state} invalid state."
        assert self.env.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        position, velocity = state
        velocity += (action - 1) * self.env.force + math.cos(3 * position) * (
            -self.env.gravity
        )
        velocity = np.clip(velocity, -self.env.max_speed, self.env.max_speed)
        position += velocity
        position = np.clip(position, self.env.min_position, self.env.max_position)
        if position == self.env.min_position and velocity < 0:
            velocity = 0

        done = bool(
            position >= self.env.goal_position and velocity >= self.env.goal_velocity
        )
        reward = -1.0

        return np.array((position, velocity), dtype=np.float32), reward, done

    def default_act(self, action):
        """Perform given action by the agent on the environment,
        and returns a reward.
        """
        self.state, reward, done, truncated, info = self.env.step(action)
        self.nb_step += 1

        return (reward, done)
