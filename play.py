from src import Game

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="name of model to load")
    parser.add_argument("--model_dir", type=str, default="models", help="path to the save directory of the models")
    args = parser.parse_args()

    game = Game(model_name=args.model_name, model_dir=args.model_dir)
    game.launch()