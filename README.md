# LP-II-

g ={
    0:[1,2],
    1:[0,3,4],
    2:[0,3],
    3:[1,2,4],
    4:[1,3]
}

def dfs(g,s):
    vis[s] = 1
    print(s)
    for c in g[s]:
        if not vis[c]:
            dfs(g,c)
            
vis = [0]*5

print("Dfs is :")
dfs(g,0)
 
print("bfs is:")
def bfs(g,s):
    q = [s]
    vis = [s]
    while q:
        cur = q.pop(0)
        print(cur)
        for c in g[cur]:
            if c not in vis:
                q.append(c)
                vis.append(c)
    
bfs(g,0)



import heapq

# Structure to represent a node in the graph
class Node:
    def __init__(self, index, g_cost, h_cost):
        self.index = index  # Index of the node
        self.g_cost = g_cost  # Cost from start node to this node
        self.h_cost = h_cost  # Heuristic cost (estimated cost from this node to goal node)
        self.f_cost = g_cost + h_cost  # f(n) = g(n) + h(n)

    # Custom comparison method for priority queue
    def __lt__(self, other):
        return self.f_cost < other.f_cost

# A* algorithm function
def AStar(graph, heuristic, start, goal):
    n = len(graph)

    # Priority queue to store nodes to be explored, ordered by f_cost
    open_list = []

    # Add the start node to the open list
    heapq.heappush(open_list, Node(start, 0, heuristic[start]))

    # Set to keep track of visited nodes
    visited = set()

    while open_list:
        # Get the node with the lowest f_cost from the open list
        current = heapq.heappop(open_list)

        # Check if the current node is the goal
        if current.index == goal:
            return current.g_cost

        # Mark current node as visited
        visited.add(current.index)

        # Expand current node
        for neighbor in range(n):
            # Check if there is a connection from current node to neighbor and neighbor is not visited
            if graph[current.index][neighbor] != 0 and neighbor not in visited:
                g_cost = current.g_cost + graph[current.index][neighbor]
                h_cost = heuristic[neighbor]
                f_cost = g_cost + h_cost

                # Add neighbor to open list
                heapq.heappush(open_list, Node(neighbor, g_cost, h_cost))

    # If goal node is not reachable
    return -1

# Main function
def main():
    n = int(input("Enter the number of nodes: "))
    graph = []

    print("Enter the adjacency matrix:")
    for _ in range(n):
        graph.append(list(map(int, input().split())))

    heuristic = list(map(int, input("Enter the heuristic values for each node: ").split()))

    start = int(input("Enter the start node: "))
    goal = int(input("Enter the goal node: "))

    shortest_path_cost = AStar(graph, heuristic, start, goal)

    if shortest_path_cost == -1:
        print(f"No path found from node {start} to node {goal}")
    else:
        print(f"Shortest path cost from node {start} to node {goal} is: {shortest_path_cost}")

if __name__ == "__main__":
    main()



03:
# A class to represent a job
class Job:
    def __init__(self, id, dead, profit):
        self.id = id  # Job Id
        self.dead = dead  # Deadline of job
        self.profit = profit  # Profit if job is over before or on deadline

# Returns maximum profit from jobs
def print_job_scheduling(arr, n):
    # Sort jobs based on their profits in descending order
    arr.sort(key=lambda x: x.profit, reverse=True)

    # Initialize result array and slots array
    result = [-1] * n
    slot = [False] * n

    # Iterate over each job
    for i in range(n):
        # Find a free slot for this job (start from the last possible slot)
        for j in range(min(n, arr[i].dead) - 1, -1, -1):
            if not slot[j]:
                result[j] = i
                slot[j] = True
                break

    # Print the job ids that are scheduled
    print("Following is the maximum profit sequence of jobs:")
    for i in range(n):
        if slot[i]:
            print(arr[result[i]].id, end=" ")


# Driver code
if __name__ == "__main__":
    # JobId, Deadline, Profit
    arr = [
        Job('a', 2, 100),
        Job('b', 1, 19),
        Job('c', 2, 27),
        Job('d', 1, 25),
        Job('e', 3, 15)
    ]

    n = len(arr)
    print_job_scheduling(arr, n)
03:
import heapq

class Solution:
    def spanningTree(self, V, adj):
        pq = [(0, 0)]  # (wt, node)
        vis = [0] * V
        total_weight = 0

        while pq:
            wt, node = heapq.heappop(pq)

            if vis[node] == 1:
                continue

            vis[node] = 1
            total_weight += wt

            for adjNode, edgWt in adj[node]:
                if not vis[adjNode]:
                    heapq.heappush(pq, (edgWt, adjNode))

        return total_weight

if __name__ == "__main__":
    V = int(input("Enter the number of vertices: "))
    adj = [[] for _ in range(V)]

    E = int(input("Enter the number of edges: "))
    print("Enter the edges in the format (source, destination, weight):")
    for _ in range(E):
        u, v, wt = map(int, input().split())
        adj[u].append((v, wt))
        adj[v].append((u, wt))  # Assuming undirected graph

    obj = Solution()
    min_spanning_tree_weight = obj.spanningTree(V, adj)
    print("Sum of weights of edges of the Minimum Spanning Tree:", min_spanning_tree_weight)




04:
from typing import List

class Solution:
    def isSafe(self, row, col, board, n):
        # Check if there is a queen in the same row to the left
        for i in range(col):
            if board[row][i] == 'Q':
                return False
        
        # Check if there is a queen in the upper left diagonal
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 'Q':
                return False
        
        # Check if there is a queen in the lower left diagonal
        for i, j in zip(range(row, n), range(col, -1, -1)):
            if board[i][j] == 'Q':
                return False
        
        return True
    
    def solve(self, col, board, ans, n):
        if col == n:
            ans.append(["".join(row) for row in board])
            return
        
        for row in range(n):
            if self.isSafe(row, col, board, n):
                board[row][col] = 'Q'
                self.solve(col + 1, board, ans, n)
                board[row][col] = '.'
    
    def solveNQueens(self, n: int) -> List[List[str]]:
        ans = []
        board = [['.' for _ in range(n)] for _ in range(n)]
        self.solve(0, board, ans, n)
        return ans

if __name__ == "__main__":
    s = Solution()
    n = int(input("Enter the n: "))
    result = s.solveNQueens(n)
    for board in result:
        for row in board:
            print(row)
        print()
