from typing import Tuple, Optional, List
from graph import Graph, Node, Edge


def dfs_hamiltonian_cycle(graph: Graph, start_node: Optional[Node] = None) -> Tuple[Graph, int, str, str]:
    """
    Находит любой гамильтонов цикл в графе с использованием алгоритма DFS.

    :param graph: Исходный граф для поиска.
    :param start_node: Опциональная стартовая вершина.
    :return: Кортеж с результатами:
        - Граф с построенным циклом
        - Суммарный вес пути
        - Статус выполнения
        - Строка пути в формате "A-B-C-A"
    """
    if not graph.nodes:
        return Graph(), 0, "Ошибка: граф пуст", ""

    if start_node is None:
        start_node = graph.nodes[0]
    elif start_node not in graph.nodes:
        return Graph(), 0, "Ошибка: неверная стартовая вершина", ""

    result_path = []
    result_weight = 0

    def backtrack(current_node: Node, path: List[Node], visited: set, weight: int) -> bool:
        nonlocal result_path, result_weight

        new_path = path + [current_node]
        new_visited = visited | {current_node}

        if len(new_visited) == len(graph.nodes):
            for edge in current_node.outgoing_edges:
                if edge.to_node == start_node:
                    result_path = new_path + [start_node]
                    result_weight = weight + edge.weight
                    return True
            return False

        for edge in current_node.outgoing_edges:
            neighbor = edge.to_node
            if neighbor not in new_visited:
                if backtrack(neighbor, new_path, new_visited, weight + edge.weight):
                    return True

        return False

    found = backtrack(start_node, [], set(), 0)

    if not found or not result_path:
        return Graph(), 0, "Гамильтонов цикл не найден", ""

    result_graph = Graph(directed=True)
    node_map = {}

    for node in result_path:
        if node.value not in node_map:
            new_node = result_graph.add_node(node.value)
            new_node.x, new_node.y = node.x, node.y
            node_map[node.value] = new_node

    total_weight = 0
    path_values = []
    for i in range(len(result_path)-1):
        from_node = result_path[i]
        to_node = result_path[i+1]

        edge = next(e for e in from_node.outgoing_edges if e.to_node == to_node)
        total_weight += edge.weight

        result_graph.add_edge(node_map[from_node.value], node_map[to_node.value], edge.weight)
        path_values.append(str(from_node.value))

    path_values.append(str(start_node.value))
    path_str = "-".join(path_values)

    return result_graph, total_weight, "Успех", path_str
