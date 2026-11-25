import math


def grid_to_world(row, col, cell_size=1.0, x0=0.0, y0=0.0):
    if not math.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("cell_size must be positive and finite")

    x = x0 + col * cell_size
    y = y0 - row * cell_size

    return x, y


def world_to_grid(x, y, cell_size=1.0, x0=0.0, y0=0.0):
    if not math.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("cell_size must be positive and finite")

    col = round((x - x0) / cell_size)
    row = round((y0 - y) / cell_size)

    return row, col


def is_at_target(x, y, target_x, target_y, tolerance=0.05):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid tolerance")

    distance = math.hypot(target_x - x, target_y - y)

    return distance <= tolerance
