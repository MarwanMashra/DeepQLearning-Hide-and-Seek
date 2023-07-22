from typing import Optional, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import time
import math
from stable_baselines3.common.env_checker import check_env
from typing import List
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import random

from src.handlers import (
    ObservationHandler, 
    MLPObservationHandler,
    CNNObservationHandler,
    MapHandler,
)

from src.utils.constants import *

class HideAndSeekEnv(gym.Env):
    def __init__(self, 
                 observation_handler: ObservationHandler,
                 map_handlers: List[MapHandler],
                 grid_size=12, 
                 max_steps=None,
                 prob_optimal_move=0.60,
                 render_mode='rgb_array',
                ):
        super(HideAndSeekEnv, self).__init__()

        assert grid_size > 0, "Grid size must be greater than 0"
        assert max_steps is None or max_steps > 0, "Max steps must be greater than 0" 
        assert render_mode in ['rgb_array', 'human'], "Render mode must be 'rgb_array' or 'human'"
        
        if max_steps is None:
            max_steps = np.inf

        self.observation_handler = observation_handler
        self.map_handlers = map_handlers
        self.map_handler = None
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.prob_optimal_move = prob_optimal_move

        self.seeker_pos = None
        self.hider_pos = None


        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right, Stay
        self.obs_dict = OBS_DICT
        self.observation_space = observation_handler.get_observation_space()

        self.current_state = None
        self.current_step = 0
        self.prev_states = None
        self.walls = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.visible_cells = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        self.current_eps = 0
        self.training = True
        self.current_eps = 0
        self.alert_1 = True
        self.alert_2 = True
        self.alert_3 = True
        self.curent_timestep = 0

        self.reset()

    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False

    def get_state(self):
        return self.observation_handler.get_observation(
            walls=self.walls, 
            visible_cells=self.visible_cells, 
            seeker_pos=self.seeker_pos, 
            hider_pos=self.hider_pos
        )
    
    def reset(self, seed: Optional[int] = None,):
        super().reset(seed=seed)

        self.current_step = 0
        self.prev_states = np.empty((self.grid_size, self.grid_size, 0), dtype=np.uint8)

        self.map_handler = random.choice(self.map_handlers)
        self.walls = self.map_handler.get_walls()
        self.seeker_pos = self.generate_seeker_pos()
        self.visible_cells = self.get_visible_cells()
        self.hider_pos = self.generate_hider_pos()

        return self.get_state(), self._get_info()

    def get_visible_cells(self):
        return self.map_handler.get_visible_cells(self.seeker_pos)
        
    def generate_hider_pos(self):
        allowed_cells = ~(self.walls | self.visible_cells)
        allowed_cells[self.seeker_pos[0], self.seeker_pos[1]] = False
        return self.sample_from_allowed_cells(allowed_cells)

    def generate_seeker_pos(self):
        allowed_cells = ~self.walls
        return self.sample_from_allowed_cells(allowed_cells)
    
    def index_to_coords(self, index):
        return index // self.grid_size, index % self.grid_size
    
    def coords_to_index(self, coords):
        return coords[0] * self.grid_size + coords[1]
    
    def sample_from_allowed_cells(self, allowed_cells):
        prob = allowed_cells.astype(np.float32)
        prob /= prob.sum()
        flat_prob = prob.flatten()
        sample_index = np.random.choice(flat_prob.size, p=flat_prob)
        x, y = self.index_to_coords(sample_index)
        return np.array([x, y], dtype=np.uint8)
    
    def _get_valid_actions(self, position):
        valid_actions = []
        if position[0] > 0 and not self.walls[position[0] - 1, position[1]]:
            valid_actions.append(UP)
        if position[0] < self.grid_size - 1 and not self.walls[position[0] + 1, position[1]]:
            valid_actions.append(DOWN)
        if position[1] > 0 and not self.walls[position[0], position[1] - 1]:
            valid_actions.append(LEFT)
        if position[1] < self.grid_size - 1 and not self.walls[position[0], position[1] + 1]:
            valid_actions.append(RIGHT)
        # valid_actions.append(STAY)
        return valid_actions
    
    def _move(self, position, action):
        if action == UP:  # Up
            position[0] -= 1
        elif action == DOWN:  # Down
            position[0] += 1
        elif action == LEFT:  # Left
            position[1] -= 1
        elif action == RIGHT:  # Right
            position[1] += 1
        return position
    
    def _get_min_distance_from_visible_cells(self, position):
        distances = np.linalg.norm(np.indices((self.grid_size, self.grid_size)) - position[:, np.newaxis, np.newaxis], axis=0)
        distances[~self.visible_cells] = np.inf
        return distances.min()
    
    def _get_info(self):
        return {}
    
    def move_player(self, action):
        assert self.action_space.contains(action), f"{action} is an invalid action"
        assert self.training == False, "You can't move the player in training mode"
        if action in self._get_valid_actions(self.seeker_pos):
            self._move(self.seeker_pos, action)
            self.visible_cells = self.get_visible_cells()
    
    def step(self, action, verbose=False):
        assert self.action_space.contains(action), f"{action} is an invalid action"

        self.current_step += 1
        self.curent_timestep += 1

        reward = 0
        reward_log = {}

        if self.training:
            # seeker makes a move based on the old state
            self._move_seeker()

        valid_actions = self._get_valid_actions(self.hider_pos)
        if action in valid_actions:
            self._move(self.hider_pos, action)
        else:
            if verbose:
                print("Hider hit the wall")
            reward += HITTING_WALL_REWARD
            reward_log['hitting_wall'] = HITTING_WALL_REWARD

        min_distance = self._get_min_distance_from_visible_cells(self.hider_pos)
        distance_reward = int(min_distance) * DISTANCE_COEF_REWARD
        reward += distance_reward
        reward_log['distance'] = distance_reward

        terminated = False
        if self.visible_cells[self.hider_pos[0], self.hider_pos[1]]:
            if verbose:
                print("Hider was caught")
            reward += LOSE_REWARD
            reward_log['lose'] = LOSE_REWARD
            terminated = True

        truncated = self.current_step >= self.max_steps and not terminated

        if terminated or truncated:
            self.current_eps += 1

        return self.get_state(), reward, terminated, truncated, self._get_info()

    def _generate_frame(self, state, cell_size=50):
        # Calculate image size based on grid dimensions and cell size
        matrix = state.reshape(self.grid_size, self.grid_size)
        image_size = (matrix.shape[1] * cell_size, matrix.shape[0] * cell_size)

        # Create a blank canvas with white background
        image = np.ones((matrix.shape[1], matrix.shape[0], 3), dtype=np.uint8) * 255

        # Fill each cell with the corresponding color using NumPy indexing
        image_rows = np.arange(matrix.shape[0]) * cell_size
        image_cols = np.arange(matrix.shape[1]) * cell_size

        # print(image[0, :60])
        image = COLORS[matrix]
        # repeat each row cell_size times
        image = np.repeat(image, cell_size, axis=0)
        image = np.repeat(image, cell_size, axis=1)

        # Draw black lines as separators between cells using NumPy indexing
        image[::cell_size, :] = (0, 0, 0)
        image[:, ::cell_size] = (0, 0, 0)
        image[::cell_size, -1] = (0, 0, 0)
        image[-1, ::cell_size] = (0, 0, 0)

        return image
    
    def render(self):
        if self.render_mode == "human":
            frame = self._generate_frame(self.get_state())
            cv2.imshow('Hide & Seek', frame)
        elif self.render_mode == "rgb_array":
            print(self.get_state())
        
    def get_best_seeker_action(self):
        return self.map_handler.get_best_seeker_action(self.seeker_pos, self.hider_pos)

    def _move_seeker(self):

        if np.random.binomial(1, self.prob_optimal_move):
            action = self.get_best_seeker_action()
        else:
            valid_actions = self._get_valid_actions(self.seeker_pos)
            action = random.choice(valid_actions)
        self._move(self.seeker_pos, action)
        self.visible_cells = self.get_visible_cells()


if __name__ == "__main__":
    observation_handler = MLPObservationHandler()
    map_handler = MapHandler()
    env = HideAndSeekEnv(observation_handler, map_handler)
    check_env(env)