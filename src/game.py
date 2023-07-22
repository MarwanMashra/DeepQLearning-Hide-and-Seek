import cv2
import os
import json

from stable_baselines3 import DQN

from src.environment import HideAndSeekEnv
from src.handlers import MLPObservationHandler, MapHandler
from src.utils.constants import *

class Game:
    def __init__(self, model_name, model_dir="models", map_name=None, map_path=None, fps=5):
        model_path = os.path.join(model_dir, model_name)
        self.model = DQN.load(os.path.join(model_path, "model.zip"))
        self.env = self._create_env(config_path=os.path.join(model_path, "config.json"), map_name=map_name, map_path=map_path)

        # put the environment in evaluation mode (not training)
        self.env.eval()
        self.fps = fps

    def launch(self):
        obs, _ = self.env.reset()
        while True:

            self.env.render()

            
            # Wait for a key press (with a small delay to limit the frame rate)
            key = cv2.waitKey(1000 // self.fps) & 0xFF

            # Map arrow keys to actions
            
            player_action = None
            if key == ASCII_ESC:  # 27 is the ASCII code for the 'ESC' key
                break
            elif key == ord('z'):  
                player_action = UP  
            elif key == ord('s'): 
                player_action = DOWN  
            elif key == ord('q'): 
                player_action = LEFT  
            elif key == ord('d'):  
                player_action = RIGHT  

            if player_action is not None:
                agent_action, _states = self.model.predict(obs, deterministic=True)
                self.env.move_player(player_action)
                obs, rewards, terminated, truncated, _ = self.env.step(agent_action)
                if terminated or truncated:
                    obs, _ = self.env.reset()

        cv2.destroyAllWindows()

    def _create_env(self, config_path, map_name, map_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        if map_name is None:
            map_name = config['map_name']
        if map_path is None:
            map_path = config['map_path']
        map_handler = MapHandler(
            grid_size=config['grid_size'],
            vision_range=config['vision_range'],
            use_cache=config['use_cache'],
            json_path=map_path,
            map_name=map_name,
        )
        grid_size = map_handler.get_grid_size()
        observation_handler = MLPObservationHandler(grid_size=grid_size)
        env = HideAndSeekEnv(
            observation_handler=observation_handler,
            map_handlers=[map_handler],
            grid_size=grid_size,
            render_mode="human",
            max_steps=config['max_steps'],
        )
        return env