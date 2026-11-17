import numpy as np


# Standard FIFA / UEFA tactical pitch dimensions in meters
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Longitudinal line Y-coordinates (meters from top/far touchline Y=0.0 to bottom/near touchline Y=68.0)
LONGITUDINAL_Y = [0.0, 13.84, 24.84, 43.16, 54.16, 68.0]

# Transverse line X-coordinates (meters from left goal line X=0.0 to right goal line X=105.0)
TRANSVERSE_X = [0.0, 5.5, 16.5, 52.5, 88.5, 99.5, 105.0]

# Complete grid of projective intersections (X_m, Y_m) of extended FIFA straight pitch lines
ALL_PROJECTIVE_INTERSECTIONS = np.array(
    [[x, y] for y in LONGITUDINAL_Y for x in TRANSVERSE_X],
    dtype=np.float64,
)


def get_screen_ref_quad(width: int, height: int) -> np.ndarray:
    """
    View-agnostic 4-corner reference quad in image pixel coordinates.
    Always lies strictly in front of the camera (z > 0) for Left, Midfield, and Right views.
    """
    return np.array(
        [
            [0.20 * width, 0.35 * height],
            [0.80 * width, 0.35 * height],
            [0.20 * width, 0.85 * height],
            [0.80 * width, 0.85 * height],
        ],
        dtype=np.float32,
    )


def build_fifa_template_points() -> np.ndarray:
    """Sample dense 2D metric points (N, 2) along all painted FIFA pitch lines, circles, and penalty arcs."""
    pts: list[list[float]] = []

    segments = [
        # Outer touchlines, goal lines & halfway line
        ((0.0, 0.0), (105.0, 0.0), 140),
        ((0.0, 68.0), (105.0, 68.0), 140),
        ((0.0, 0.0), (0.0, 68.0), 90),
        ((52.5, 0.0), (52.5, 68.0), 90),
        ((105.0, 0.0), (105.0, 68.0), 90),
        # Left 18-yard penalty area
        ((0.0, 13.84), (16.5, 13.84), 35),
        ((0.0, 54.16), (16.5, 54.16), 35),
        ((16.5, 13.84), (16.5, 54.16), 60),
        # Left 6-yard goal area
        ((0.0, 24.84), (5.5, 24.84), 15),
        ((0.0, 43.16), (5.5, 43.16), 15),
        ((5.5, 24.84), (5.5, 43.16), 28),
        # Right 18-yard penalty area
        ((88.5, 13.84), (105.0, 13.84), 35),
        ((88.5, 54.16), (105.0, 54.16), 35),
        ((88.5, 13.84), (88.5, 54.16), 60),
        # Right 6-yard goal area
        ((99.5, 24.84), (105.0, 24.84), 15),
        ((99.5, 43.16), (105.0, 43.16), 15),
        ((99.5, 24.84), (99.5, 43.16), 28),
    ]
    for (x1, y1), (x2, y2), count in segments:
        for t in np.linspace(0.0, 1.0, count):
            pts.append([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])

    # Center circle: center (52.5, 34.0), radius 9.15m
    for ang in np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False):
        pts.append([52.5 + 9.15 * np.cos(ang), 34.0 + 9.15 * np.sin(ang)])

    # Left penalty arc: center (11.0, 34.0), radius 9.15m, outside penalty area (X >= 16.5)
    for ang in np.linspace(-0.5 * np.pi, 0.5 * np.pi, 50):
        px = 11.0 + 9.15 * np.cos(ang)
        py = 34.0 + 9.15 * np.sin(ang)
        if px >= 16.5:
            pts.append([px, py])

    # Right penalty arc: center (94.0, 34.0), radius 9.15m, outside penalty area (X <= 88.5)
    for ang in np.linspace(0.5 * np.pi, 1.5 * np.pi, 50):
        px = 94.0 + 9.15 * np.cos(ang)
        py = 34.0 + 9.15 * np.sin(ang)
        if px <= 88.5:
            pts.append([px, py])

    return np.asarray(pts, dtype=np.float64)
