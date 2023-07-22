from src import Game

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="best_model", help="name of model to load")
    parser.add_argument("--model_dir", type=str, default="models", help="path to the save directory of the models")
    parser.add_argument("--map_name", type=str, default="default", help="name of the map to load")
    parser.add_argument("--map_path", type=str, default="src/maps.json", help="path to the map to load")
    parser.add_argument("--fps", type=int, default=5, help="frames per second")
    
    args = parser.parse_args()

    game = Game(model_name=args.model_name, model_dir=args.model_dir, map_name=args.map_name, map_path=args.map_path, fps=args.fps)
    game.launch()