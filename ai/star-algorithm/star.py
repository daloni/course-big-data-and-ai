import pyamaze as maze
import time
from queue import PriorityQueue
from math import sqrt

ROWS = 20
COLS = 20

def distance(cell1, cell2):
    return sqrt((cell2[0] - cell1[0]) ** 2 + (cell2[1] - cell1[1]) ** 2)

def aStar(m, init_position, goal_position):
    # Save positions with the a* value => positions
    forward_path = PriorityQueue()
    position = init_position
    point = m.maze_map[init_position]
    extended_list = {}
    aux = 0

    while aux < 5:
        distance_from_init = 0
        distance_to_goal = distance(init_position, goal_position)
        total_distance = distance_from_init + distance_to_goal
        pq.put(total_distance)

        print(total_distance, point)
        x, y = position

        validMovements = sum(point[index] for index in point)
        newX = x
        newY = y

        for i in range(validMovements):
            if point['E']:
                if (newX, newY + 1) not in extended_list:
                    newY = y + 1
                    print('Move to east', newX, newY)
            elif point['W']:
                if (newX, newY - 1) not in extended_list:
                    newY = y - 1
                    print('Move to west', newX, newY)
            elif point['N']:
                if (newX + 1, newY) not in extended_list:
                    newX = x + 1
                    print('Move to north', newX, newY)
            elif point['S']:
                if (newX - 1, newY) not in extended_list:
                    newX = x - 1
                    print('Move to south', newX, newY)

        newPosition = (newX, newY)
        extended_list[newPosition] = True
        position = newPosition

        aux = aux + 1

    return forward_path

m=maze.maze(ROWS,COLS)
m.CreateMaze()
a=maze.agent(m,footprints=True)

pre_Astar = time.time()
path = aStar(m, a.position, a.goal)
post_Astar = time.time()

# print(post_Astar - pre_Astar)

m.tracePath({a:path},delay=5)
m.run()
