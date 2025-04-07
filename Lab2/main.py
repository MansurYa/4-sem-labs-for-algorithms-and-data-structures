from colorGUI import LabApp
from graph import Graph
import random


def create_graph(num_nodes, num_edges, canvas_width, canvas_height):
    """
    Создаёт граф с заданным количеством вершин и рёбер

    :param num_nodes: количество вершин
    :param num_edges: количество рёбер
    :param canvas_width: ширина холста
    :param canvas_height: высота холста
    :return: созданный граф
    """
    graph = Graph(directed=True)

    cols = int(num_nodes**0.5)
    rows = (num_nodes + cols - 1) // cols
    padding = 30

    x_step = (canvas_width - 2*padding) / (cols-1) if cols > 1 else 0
    y_step = (canvas_height - 2*padding) / (rows-1) if rows > 1 else 0

    nodes = []
    for i in range(num_nodes):
        col = i % cols
        row = i // cols
        x = padding + col * x_step + random.randint(-10, 10)
        y = padding + row * y_step + random.randint(-10, 10)
        node = graph.add_node(i)
        node.x = x
        node.y = y
        nodes.append(node)

    all_possible_edges = [(i, j) for i in range(num_nodes)
                         for j in range(num_nodes) if i != j]
    selected_edges = random.sample(all_possible_edges, num_edges)

    for from_idx, to_idx in selected_edges:
        weight = random.randint(1, 50)
        graph.add_edge(nodes[from_idx], nodes[to_idx], weight)

    return graph

def main():
    # graph = create_graph(num_nodes=2, num_edges=1, canvas_width=335, canvas_height=190)
    graph = create_graph(num_nodes=50, num_edges=1000, canvas_width=678, canvas_height=395)
    # graph = create_graph(num_nodes=100, num_edges=6000, canvas_width=678, canvas_height=395)
    # graph = Graph()

    app = LabApp(graph)
    app.attributes('-fullscreen', True)
    app.mainloop()


if __name__ == "__main__":
    main()
