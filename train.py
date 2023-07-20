from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from src.environment import HideAndSeekEnv
from src.handlers import MLPObservationHandler, MapHandler

import argparse

def train(args):

    observation_handler = MLPObservationHandler(grid_size=args.grid_size)
    map_handler = MapHandler(
        grid_size=args.grid_size,
        vision_range=args.vision_range,
        use_cache=args.use_cache,
        json_path=args.map_path,
        map_name=args.map_name,
    )

    env = Monitor(HideAndSeekEnv(
        observation_handler=observation_handler,
        map_handler=map_handler,
        grid_size=args.grid_size,
        render_mode="rgb_array",
        max_steps=args.max_steps,
    ))

    model = DQN("MlpPolicy", env, 
                verbose=0,
                learning_rate=args.lr,
                exploration_final_eps=args.exp,
                tensorboard_log=args.log_dir,
                )
    
    model.learn(
        total_timesteps=args.timesteps, 
        progress_bar=args.progress_bar,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_path", type=str, default="maps.json")
    parser.add_argument("--map_name", type=str, default="default")
    parser.add_argument("--vision_range", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=12)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--render_mode", type=str, default="human")
    parser.add_argument("--use_cache", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--exp", type=float, default=0.1)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--timesteps", type=str, default=500_000)
    parser.add_argument("--progress_bar", type=bool, default=True)

    args = parser.parse_args()

    train(args)