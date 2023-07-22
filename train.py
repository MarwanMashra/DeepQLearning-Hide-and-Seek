from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from src import HideAndSeekEnv
from src.handlers import MLPObservationHandler, MapHandler

import argparse
import datetime
import json
import os

def train(args):

    observation_handler = MLPObservationHandler(grid_size=args.grid_size)
    map_handlers = []
    for map_name in args.map_names:
        map_handlers.append(
            MapHandler(
            grid_size=args.grid_size,
            vision_range=args.vision_range,
            use_cache=args.use_cache,
            json_path=args.map_path,
            map_name=map_name,
            )
        )

    env = Monitor(HideAndSeekEnv(
        observation_handler=observation_handler,
        map_handlers=map_handlers,
        grid_size=args.grid_size,
        render_mode="rgb_array",
        max_steps=args.max_steps,
        prob_optimal_move=args.prob_optimal_move,
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
    if args.save_model:
        os.makedirs(args.model_dir, exist_ok=True)
        model_name = args.model_name
        if model_name is None:
            now = datetime.datetime.now()
            date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
            model_name = f"{date_string}"
        model_path = os.path.join(args.model_dir, model_name)
        os.makedirs(model_path)
        model.save(os.path.join(model_path, "model.zip"))
        with open(os.path.join(model_path, "config.json"), "w") as f:
            json.dump(vars(args), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_path", type=str, default="src/maps.json")
    parser.add_argument("--map_names", type=str, nargs="+", default=["default", "map1", "map2", "map3", "map4"])
    parser.add_argument("--vision_range", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=12)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--use_cache", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--exp", type=float, default=0.1)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--timesteps", type=str, default=500_000)
    parser.add_argument("--progress_bar", type=bool, default=True)
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--prob_optimal_move", type=float, default=0.60)

    args = parser.parse_args()

    train(args)