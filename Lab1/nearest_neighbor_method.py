from graph import Graph, Edge
import random
from typing import List, Tuple


def clear_edge_with_visited_node(edges: List[Edge]) -> List[Edge]:
    """Фильтрует рёбра, исключая те, которые ведут в посещённые узлы."""
    return [edge for edge in edges if not edge.to_node.visited]


def nearest_neighbor_method(graph: Graph, start_node=None) -> Tuple[Graph, int, str, str]:
    """
    Реализация алгоритма ближайшего соседа для поиска гамильтонова цикла

    :param graph: Исходный взвешенный орграф для анализа.
    :param start_node: Стартовая вершина
    :return: Кортеж с результатами:
            - Граф с построенным циклом (или частичным путем)
            - Суммарный вес найденного пути
            - Статус выполнения (успех/ошибка)
            - Строка пути в формате "0-4-3-1-2-0"
    """
    total_weight = 0
    result_graph = Graph(directed=True)

    for node in graph.nodes:
        node.visited = False

    if not graph.nodes:
        return Graph(), 0, "Ошибка: граф пуст.", ""

    try:
        if start_node is None:
            start_node = random.choice(graph.nodes) if graph.nodes else None
        else:
            if start_node not in graph.nodes:
                return Graph(), 0, "Ошибка: неверная стартовая вершина", ""

        current_node = start_node
        current_node.visited = True
        path = [str(current_node.value)]

        # start_node = random.choice(graph.nodes)
        # current_node = start_node
        # current_node.visited = True
        # path.append(str(current_node.value))

        node_mapping = {}
        start_result_node = result_graph.add_node(start_node.value)
        start_result_node.x, start_result_node.y = start_node.x, start_node.y
        node_mapping[start_node.value] = start_result_node
        previous_result_node = start_result_node

        while len(result_graph.nodes) < len(graph.nodes):
            available_edges = clear_edge_with_visited_node(current_node.sorted_edges())
            if not available_edges:
                message = "Ошибка: Нет доступных узлов для продолжения."
                return result_graph, total_weight, message, "-".join(path)

            next_edge = available_edges[0]
            total_weight += next_edge.weight
            next_node = next_edge.to_node
            next_node.visited = True
            path.append(str(next_node.value))

            if next_node.value not in node_mapping:
                new_node = result_graph.add_node(next_node.value)
                new_node.x, new_node.y = next_node.x, next_node.y
                node_mapping[next_node.value] = new_node
            result_graph.add_edge(previous_result_node, node_mapping[next_node.value], next_edge.weight)
            previous_result_node = node_mapping[next_node.value]
            current_node = next_node

        closing_edges = [edge for edge in current_node.outgoing_edges if edge.to_node == start_node]
        if not closing_edges:
            message = "Ошибка: Нет ребра для замыкания цикла."
            return result_graph, total_weight, message, "-".join(path)

        closing_edge = min(closing_edges, key=lambda e: e.weight)
        total_weight += closing_edge.weight
        result_graph.add_edge(previous_result_node, node_mapping[start_node.value], closing_edge.weight)
        path.append(str(start_node.value))

        message = "Гамильтонов цикл успешно найден."
        return result_graph, total_weight, message, "-".join(path)

    finally:
        for node in graph.nodes:
            node.visited = False
