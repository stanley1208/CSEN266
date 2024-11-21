edges = [
    ("Arad", "Zerind", 75),
    ("Arad", "Sibiu", 140),
    ("Arad", "Timisoara", 118),
    ("Bucharest", "Pitesti", 101),
    ("Bucharest", "Urziceni", 85),
    ("Bucharest", "Giurglu", 90),
    ("Craiova", "Dobreta", 120),
    ("Fagaras", "Bucharest", 211),
    ("Hirsova", "Eforie", 86),
    ("Lasi", "Neamt", 87),
    ("Lugoj", "Mehadia", 70),
    ("Mehadia", "Dobreta", 75),
    ("Oradea", "Sibiu", 151),
    ("Pitesti", "Craiova", 138),
    ("Rimnicu Vilcea", "Pitesti", 97),
    ("Rimnicu Vilcea", "Craiova", 146),
    ("Sibiu", "Fagaras", 99),
    ("Sibiu", "Rimnicu Vilcea", 80),
    ("Timisoara", "Lugoj", 111),
    ("Urziceni", "Hirsova", 98),
    ("Urziceni", "Vaslui", 142),
    ("Vaslui", "Lasi", 92),
    ("Zerind", "Oradea", 71)
]


graph = {}

for edge in edges:
    node1, node2, weight = edge

    if node1 not in graph:
        graph[node1] = []
    if node2 not in graph:
        graph[node2] = []


    graph[node1].append((node2, weight))
    graph[node2].append((node1, weight))

# Count vertices and edges
nodes = len(graph)
edges = len(edges)

# Print results
print("Number of vertices:", nodes)
print("Number of edges:", edges)
