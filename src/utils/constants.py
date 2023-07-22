import numpy as np

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

HITTING_WALL_REWARD = -10
LOSE_REWARD = -100
DISTANCE_COEF_REWARD = 1

ASCII_LEFT = 81
ASCII_RIGHT = 83
ASCII_UP = 82
ASCII_DOWN = 84
ASCII_ESC = 27

OBS_DICT = {
            "hidden":0,
            "visible":1,
            "wall":2,
            "seeker":3,
            "hider":4
        }

MIN_CNN_RES = 36


GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)

COLORS = np.array([WHITE,  # hidden cells
                   YELLOW, # visible cells
                   BLACK,  # walls
                   RED,  # seeker
                   GREEN],   # hider
                  dtype=np.uint8)
