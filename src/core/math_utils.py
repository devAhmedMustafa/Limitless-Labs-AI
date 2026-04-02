import numpy as np


def calc_angle(a, b, c):
    """Return the joint angle in degrees for points a-b-c."""
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    c_arr = np.asarray(c, dtype=np.float32)
    ba = a_arr - b_arr
    bc = c_arr - b_arr
    cross = ba[0] * bc[1] - ba[1] * bc[0]
    dot = float(np.dot(ba, bc))
    angle = np.degrees(np.arctan2(abs(cross), dot))
    return float(angle)
