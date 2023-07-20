import numpy as np

def compute_visible_cells(walls, agent_position, distance):
        grid_size = walls.shape[0]
        visible_cells = np.zeros((grid_size, grid_size), dtype=bool)

        x, y = agent_position

        left_limit = max(0, x - distance)
        right_limit = min(grid_size, x + distance + 1)
        top_limit = max(0, y - distance)
        bottom_limit = min(grid_size, y + distance + 1)

        for new_x in range(left_limit, right_limit):
            for new_y in range(top_limit, bottom_limit):

                if not (new_x==x and new_y==y) and walls[new_x, new_y] == 0:
                    if has_line_of_sight(walls, (x, y), (new_x, new_y)):
                        visible_cells[new_x, new_y] = True

        visible_cells[x, y] = False

        return visible_cells


def has_line_of_sight(walls, a1, a2):
    # Get the dimensions of the matrix
    intermediate_cells = bresenham(a1, a2)
    # print(intermediate_cells.shape)
    is_wall = walls[intermediate_cells[:, 0], intermediate_cells[:, 1]]
    return np.all(~is_wall)

# from https://github.com/encukou/bresenham/blob/master/bresenham.py
def bresenham(p1, p2):
    x0, y0 = p1
    x1, y1 = p2
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).

    Input coordinates should be integers.

    The result will contain both the start and the end point.
    """
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0
    result = []
    for x in range(dx + 1):
        
        result.append(np.array([x0 + x * xx + y * yx, y0 + x * xy + y * yy], dtype=np.uint8))

        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy 
    return np.array(result, dtype=np.uint8)
