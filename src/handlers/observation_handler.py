from abc import ABC, abstractmethod
from gymnasium import spaces
import numpy as np
from src.utils.constants import *

class ObservationHandler(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        pass

    @abstractmethod
    def get_observation(self,
            walls: np.ndarray, 
            visible_cells: np.ndarray, 
            seeker_pos: np.ndarray, 
            hider_pos: np.ndarray
        ) -> np.ndarray:
        pass

    def to_state(self, 
            walls: np.ndarray, 
            visible_cells: np.ndarray, 
            seeker_pos: np.ndarray, 
            hider_pos: np.ndarray
        ) -> np.ndarray:
        
        state = np.full(
            shape=(self.grid_size, self.grid_size), 
            fill_value=OBS_DICT['hidden'], 
            dtype=np.uint8
        )
        state[visible_cells] = OBS_DICT['visible']
        state[walls] = OBS_DICT['wall']
        state[seeker_pos[0], seeker_pos[1]] = OBS_DICT['seeker']
        state[hider_pos[0], hider_pos[1]] = OBS_DICT['hider']
        return state

class MLPObservationHandler(ObservationHandler):
    def __init__(self, grid_size: int = 12) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.observation_space = spaces.Box(
            low=0,
            high=len(OBS_DICT)-1,
            shape=(self.grid_size*self.grid_size, ),
            dtype=np.uint8
        )

    def get_observation_space(self) -> spaces.Space:
        return self.observation_space

    def get_observation(self, 
            walls: np.ndarray, 
            visible_cells: np.ndarray, 
            seeker_pos: np.ndarray, 
            hider_pos: np.ndarray
        ) -> np.ndarray:
        
        current_state = self.to_state(walls, visible_cells, seeker_pos, hider_pos)
        return current_state.flatten()
        
class CNNObservationHandler(ObservationHandler):
    def __init__(self, grid_size: int = 12) -> None:
        raise NotImplementedError