def main():
    import heapq
    import matplotlib.pyplot as plt

    graph = {
        "A": {"B": 2, "C": 5},
        "B": {"D": 2, "E": 4},
        "C": {"G": 6},
        "D": {"G": 4},
        "E": {"G": 1},
        "G": {}
    }

    def ucs(start, goal):
        queue = [(0, start, [start])]
        visited = set()
        while queue:
            cost, node, path = heapq.heappop(queue)
            if node == goal:
                return cost, path
            if node in visited:
                continue
            visited.add(node)
            for neighbour, weight in graph[node].items():
                if neighbour not in visited:
                    heapq.heappush(queue, (
                        cost + weight, neighbour, path + [neighbour]
                    ))
        return None

    result = ucs("A", "G")
    if result is None:
        print("Path not found")
        return
    cost, path = result
    print("Minimum Cost:", cost)
    print("Optimal Path:", " -> ".join(path))

    pos = {
        "A": (0, 1), "B": (1, 2), "C": (1, 0),
        "D": (2, 2), "E": (2, 1), "G": (3, 1)
    }
    chosen_edges = set(zip(path, path[1:]))
    for a, neighbours in graph.items():
        for b, weight in neighbours.items():
            x1, y1 = pos[a]
            x2, y2 = pos[b]
            chosen = (a, b) in chosen_edges
            plt.plot([x1, x2], [y1, y2],
                     color="green" if chosen else "grey",
                     linewidth=3 if chosen else 1)
            plt.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, str(weight))
    for node, (x, y) in pos.items():
        plt.scatter(x, y, s=600, color="steelblue")
        plt.text(x, y, node, ha="center", va="center", color="white")
    plt.title("Uniform Cost Search - Optimal Path")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
