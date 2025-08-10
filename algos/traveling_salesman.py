
import numpy as np
from numba import njit
import numpy as np

@njit
def distance_numba(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


@njit(nogil=True)
def approach_greedy(points, num_points, progress_proxy):
    path = [0]
    used = np.zeros(num_points, dtype=np.bool_)
    used[0] = True
    for _ in range(1, num_points):
        last = path[-1]
        min_dist = 1e9
        next_index = -1
        for i in range(num_points): 
            if not used[i]:
                d = distance_numba(points[last], points[i])
                if d < min_dist:
                    min_dist = d
                    next_index = i

            progress_proxy.update(1)

        path.append(next_index)
        used[next_index] = True
    return path

