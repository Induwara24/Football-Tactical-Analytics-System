import math


def get_center_of_bbox(bbox: list | tuple) -> tuple[int, int]:
    """Return integer (cx, cy) center of an [x1, y1, x2, y2] bounding box."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox: list | tuple) -> int:
    """Return integer pixel width of an [x1, y1, x2, y2] bounding box."""
    return int(bbox[2] - bbox[0])


def get_foot_position(bbox: list | tuple) -> tuple[int, int]:
    """Return bottom-center (x_center, y_bottom) foot coordinate for 2D pitch projection."""
    x1, _, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def measure_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Return Euclidean distance between two 2D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def measure_xy_distance(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float]:
    """Return signed (dx, dy) displacement between p1 and p2."""
    return p1[0] - p2[0], p1[1] - p2[1]

