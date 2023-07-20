import json
import numpy as np
from src.utils.constants import *
from src.utils.vision import compute_visible_cells

from pathfinding.core.grid import Grid
from pathfinding.finder.dijkstra import DijkstraFinder

class MapHandler:
    def __init__(self, 
                 grid_size=12, 
                 vision_range=5,
                 use_cache=True, 
                 json_path="maps.json",
                 map_name="default"
                ):

        assert vision_range > 0, "Vision range must be greater than 0"

        self.grid_size = grid_size
        self.use_cache = use_cache
        self.json_path = json_path
        self.map_name = map_name
        self.vision_range = vision_range

        self.walls = self._load_map()

        self.cache = {
            'best_seeker_action':-1*np.ones(
                (self.grid_size, self.grid_size, self.grid_size, self.grid_size), 
                dtype=int,
            ),
            'visible_cells':{},
        }

    def _load_map(self):
        with open(self.json_path, "r") as f:
            maps = json.load(f)
        walls = maps[self.map_name]

        assert len(walls) == self.grid_size and len(walls[0]) == self.grid_size, "Map size is not equal to grid size"
        
        walls = [[c == 'x' for c in row] for row in walls]
        return np.array(walls)

    def get_walls(self):
        return self.walls.copy()
    
    def get_grid_size(self):
        return self.grid_size

    def get_visible_cells(self, seeker_pos):
        cache = self.cache['visible_cells']
        if self.use_cache:
            key = (seeker_pos[0], seeker_pos[1])
            if key in cache:
                return cache[key]

        visible_cells = compute_visible_cells(self.walls, seeker_pos, self.vision_range)
        if self.use_cache:
            cache[key] = visible_cells

        return visible_cells

    def get_best_seeker_action(self, seeker_pos, hider_pos):
        cache = self.cache['best_seeker_action']
        if self.use_cache:
            key = (seeker_pos[0], seeker_pos[1], hider_pos[0], hider_pos[1])
            if cache[key]!=-1:
                return cache[key]

        best_action = self._compute_best_seeker_action(seeker_pos, hider_pos)
        if self.use_cache:
            cache[key] = best_action

        return best_action

    def _compute_best_seeker_action(self, seeker_pos, hider_pos):
        grid = Grid(matrix=~self.walls)
        start = grid.node(seeker_pos[1], seeker_pos[0])
        end = grid.node(hider_pos[1], hider_pos[0])
        finder = DijkstraFinder()
        path, runs = finder.find_path(start, end, grid)
        next_cell = path[-len(path)+1]
        best_move = (next_cell[1] - seeker_pos[0], next_cell[0] - seeker_pos[1])
        best_action = self._move_to_action(best_move)
        return best_action

    def _move_to_action(self, move):
        if move == (-1, 0):
            action = UP
        elif move == (1, 0):
            action = DOWN
        elif move == (0, -1):
            action = LEFT
        elif move == (0, 1):
            action = RIGHT
        return action
