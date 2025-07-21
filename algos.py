import cv2
import numpy as np
import math
from scipy.spatial import KDTree
import time
from numba import njit
from numba_progress import ProgressBar
from tqdm import tqdm
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from tqdm import tqdm
import numpy as np


def distance(p1, p2):
    # Euclidean distance in 2D
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

@njit
def distance_numba(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def greedy_path(points):
    path = [points[0]]
    used = [False] * len(points)
    used[0] = True
    for idx in range(1, len(points)):
        last = path[-1]
        next_index = min((i for i in range(len(points)) if not used[i]), key=lambda i: distance(last, points[i]))
        path.append(points[next_index])
        used[next_index] = True
        if idx % 100 == 0:
            print(f"Progress: {idx}/{len(points)}")
    return path

def fast_greedy_path(points):
    points = np.array(points)
    tree = KDTree(points)
    used = np.zeros(len(points), dtype=bool)
    path = [0]
    used[0] = True
    for _ in range(1, len(points)):
        dist, idx = tree.query(points[path[-1]], k=len(points))
        for i in idx:
            if not used[i]:
                path.append(i)
                used[i] = True
                break
        if _ % 100 == 0:
            print(f"Progress: {_}/{len(points)}")
    return points[path]


@njit(nogil=True)
def greedy_path_numba_pb(points, num_points, progress_proxy):
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

@njit(nogil=True)
def greedy_path_numba(points, num_points):
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
        path.append(next_index)
        used[next_index] = True
    return path


def clustered_greedy_tsp(points, n_clusters):
    points = np.array(points)

    # cluster the points
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(points)

    final_path = []

    for cluster_id in tqdm(range(n_clusters), desc="Solving cluster paths"):
        cluster_pts  = points[labels == cluster_id]
        if len(cluster_pts ) <= 2:
            final_path.extend(cluster_pts .tolist())
            continue

        # Solve greedy TSP for this cluster
        idx_path = greedy_path_numba(cluster_pts, len(cluster_pts))
        ordered_points = [tuple(cluster_pts [i]) for i in idx_path]
        final_path.extend(ordered_points)

    return final_path