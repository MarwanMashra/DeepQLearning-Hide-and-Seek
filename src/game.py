import cv2
import os
import json

from stable_baselines3 import DQN

from src.environment import HideAndSeekEnv
from src.handlers import MLPObservationHandler, MapHandler
from src.utils.constants import *

class Game:
    def __init__(self, model_name, model_dir="models", map_name="default", map_path="maps.json"):
        model_path = os.path.join(model_dir, model_name)
        self.model = DQN.load(os.path.join(model_path, "model.zip"))
        self.env = self._create_env(config_path=os.path.join(model_path, "config.json"))

        # put the environment in evaluation mode (not training)
        self.env.eval()

    def launch(self):
        obs, _ = self.env.reset()
        while True:

            self.env.render()

            
            # Wait for a key press (with a small delay to limit the frame rate)
            key = cv2.waitKey(1000 // self.env.fps) & 0xFF

            # Map arrow keys to actions
            
            player_action = None
            if key == ASCII_ESC:  # 27 is the ASCII code for the 'ESC' key
                break
            elif key == ord('z'):  # ASCII code for the 'up' arrow key
                player_action = UP  # Assuming action 0 corresponds to moving 'up'
            elif key == ord('s'):  # ASCII code for the 'down' arrow key
                player_action = DOWN  # Assuming action 1 corresponds to moving 'down'
            elif key == ord('q'):  # ASCII code for the 'left' arrow key
                player_action = LEFT  # Assuming action 2 corresponds to moving 'left'
            elif key == ord('d'):  # ASCII code for the 'right' arrow key
                player_action = RIGHT  # Assuming action 3 corresponds to moving 'right'

            if player_action is not None:
                agent_action, _states = self.model.predict(obs, deterministic=True)
                self.env.move_player(player_action)
                obs, rewards, terminated, truncated, _ = self.env.step(agent_action)
                if terminated or truncated:
                    obs, _ = self.env.reset()

    def _create_env(self, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        observation_handler = MLPObservationHandler(grid_size=config['grid_size'])
        map_handler = MapHandler(
            grid_size=config['grid_size'],
            vision_range=config['vision_range'],
            use_cache=config['use_cache'],
            json_path=config['map_path'],
            map_name=config['map_name'],
        )
        env = HideAndSeekEnv(
            observation_handler=observation_handler,
            map_handler=map_handler,
            grid_size=config['grid_size'],
            render_mode="human",
            max_steps=config['max_steps'],
        )
        return env
    
# cv2.waitKey(0)
# cv2.destroyAllWindows()