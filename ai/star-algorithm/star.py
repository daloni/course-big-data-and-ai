import pyamaze as maze
import time
from queue import PriorityQueue
from math import sqrt

ROWS = 100
COLS = 100
POSSIBLE_PATHS = 10 # 0 => only 1 path | 99 (max) => multiple paths

# Euclidean distance
def distanceEuclidean(cell1, cell2):
    return sqrt((cell2[0] - cell1[0]) ** 2 + (cell2[1] - cell1[1]) ** 2)

# Manhattan distance
def distanceManhattan(cell1, cell2):
    return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])    

def distance(cell1, cell2):
    return distanceManhattan(cell1, cell2)

def aStar(m, a):
    start_position = a.position
    goal_position = a.goal

    g_score = { cell: float('inf') for cell in m.grid }
    f_score = { cell: float('inf') for cell in m.grid }

    g_score[start_position] = 0
    f_score[start_position] = distance(start_position, goal_position)

    forward_path = {}
    extended_list = {}

    open = PriorityQueue()
    open.put((distance(start_position, goal_position), distance(start_position, goal_position), start_position))

    while not open.empty():
        current_cel = open.get()[2]

        if current_cel == goal_position:
            break

        for direction in 'ESNW':
            if m.maze_map[current_cel][direction] != True:
                continue

            if direction == 'E':
                child_cel = (current_cel[0], current_cel[1] + 1)
            if direction == 'W':
                child_cel = (current_cel[0], current_cel[1] - 1)
            if direction == 'N':
                child_cel = (current_cel[0] - 1, current_cel[1])
            if direction == 'S':
                child_cel = (current_cel[0] + 1, current_cel[1])

            current_g_score = g_score[current_cel] + 1
            current_f_score = current_g_score + distance(child_cel, goal_position)

            if current_f_score < f_score[child_cel]:
                g_score[child_cel] = current_g_score
                f_score[child_cel] = current_f_score

                open.put((current_f_score, distance(child_cel, goal_position), child_cel))
                extended_list[child_cel] = current_cel

    cell = goal_position

    while cell != start_position:
        forward_path[extended_list[cell]] = cell
        cell = extended_list[cell]

    return forward_path

m=maze.maze(ROWS, COLS)
m.CreateMaze(loopPercent=POSSIBLE_PATHS)
a=maze.agent(m, footprints=True)

pre_Astar = time.time()
path = aStar(m, a)
post_Astar = time.time()

print(post_Astar - pre_Astar)

m.tracePath({ a: path }, delay=5)
m.run()
