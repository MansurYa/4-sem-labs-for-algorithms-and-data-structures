from graph import Graph, Node, Edge
from typing import Tuple, Optional, List
import random
import math
from DFS import dfs_hamiltonian_cycle


def calculate_path_length(graph: Graph, path: List[Node]) -> int:
    """
    Вычисляет длину гамильтонова цикла.

    :param graph: Исходный граф
    :param path: Список узлов пути
    :return: Длина пути или бесконечность при невалидном пути
    """
    if len(path) != len(graph.nodes) + 1 or path[0] != path[-1]:
        return float('inf')

    total = 0
    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i + 1]

        # Находим соответствующие вершины в исходном графе
        original_from_node = next(n for n in graph.nodes if n.value == from_node.value)
        original_to_node = next(n for n in graph.nodes if n.value == to_node.value)

        # Ищем ребро в исходном графе
        edge = next(
            (e for e in original_from_node.outgoing_edges if e.to_node == original_to_node),
            None
        )

        if not edge:
            return float('inf')

        total += edge.weight

    return total


def get_initial_path(graph: Graph) -> List[Node]:
    """
    Генерирует начальный путь через DFS.

    :param graph: Исходный граф
    :return: Список узлов начального пути
    :raises ValueError: Если DFS не нашел цикл
    """
    result_graph, _, status, path_str = dfs_hamiltonian_cycle(graph)

    if status != "Успех":
        raise ValueError("DFS не нашёл начальный цикл")

    path = []
    for val in path_str.split("-")[:-1]:
        original_node = next(n for n in graph.nodes if str(n.value) == val)
        node_copy = Node(original_node.value)
        node_copy.x = original_node.x
        node_copy.y = original_node.y
        node_copy.outgoing_edges = original_node.outgoing_edges
        path.append(node_copy)

    path.append(path[0])
    return path


def two_opt_swap(path: List[Node], i: int, k: int) -> List[Node]:
    """
    Выполняет 2-opt swap для генерации соседнего решения.

    :param path: Исходный путь
    :param i: Начальный индекс
    :param k: Конечный индекс
    :return: Новый путь после перестановки
    """
    new_path = path.copy()
    new_path[i:k+1] = reversed(new_path[i:k+1])

    for idx in range(len(new_path)):
        new_path[idx].x = path[idx].x
        new_path[idx].y = path[idx].y

    return new_path


def simulated_annealing(
    graph: Graph,
    initial_temp: Optional[float] = None,
    min_temp: float = 1,
    cooling_rate: float = 0.95,
    max_iterations: Optional[int] = None,
    fast_annealing: bool = False
) -> Tuple[Graph, int, str, str]:
    """
    Реализация алгоритма имитации отжига с DFS-инициализацией.

    :param graph: Входной ориентированный граф
    :param initial_temp: Начальная температура (автовычисление)
    :param min_temp: Минимальная температура
    :param cooling_rate: Коэффициент охлаждения
    :param max_iterations: Число итераций (автовычисление)
    :param fast_annealing: Флаг сверхбыстрого отжига
    :return: Граф, длина пути, статус, строка пути
    """
    try:
        current_path = get_initial_path(graph)
    except ValueError as e:
        return Graph(), 0, f"Ошибка инициализации: {str(e)}", ""

    current_length = calculate_path_length(graph, current_path)
    if current_length == float('inf'):
        return Graph(), 0, "Невалидный начальный путь", ""

    best_path = current_path.copy()
    best_length = current_length
    temp = initial_temp if initial_temp is not None else 100.0  # Дефолтное значение
    num_of_iteration = 0

    while temp >= min_temp:
        num_of_iteration += 1
        print(f"Итерация: {num_of_iteration}, температура: {temp}, текущая длина: {current_length}")

        a, b = sorted(random.sample(range(1, len(current_path)-1), 2))
        new_path = two_opt_swap(current_path, a, b)
        new_length = calculate_path_length(graph, new_path)

        if new_length == float('inf'):
            continue

        delta = new_length - current_length
        accept_prob = math.exp(-delta / (temp * (1 + 0.1 * current_length / best_length)))

        if delta < 0 or random.random() < accept_prob:
            current_path = new_path
            current_length = new_length

            if current_length < best_length:
                best_path = current_path.copy()
                best_length = current_length

        if fast_annealing:
            temp = initial_temp / (1 + num_of_iteration) if initial_temp is not None else 100.0 / (1 + num_of_iteration)
        else:
            temp *= cooling_rate

    result_graph = Graph(directed=True)
    node_map = {}

    for node in graph.nodes:
        new_node = result_graph.add_node(node.value)
        new_node.x = node.x
        new_node.y = node.y
        node_map[node.value] = new_node

    total_weight = 0
    path_values = []
    for i in range(len(best_path)-1):
        from_val = best_path[i].value
        to_val = best_path[i+1].value

        original_from = next(n for n in graph.nodes if n.value == from_val)
        original_to = next(n for n in graph.nodes if n.value == to_val)

        edge = next((e for e in original_from.outgoing_edges if e.to_node == original_to), None)
        if not edge:
            return Graph(), 0, "Ошибка построения пути", ""

        total_weight += edge.weight
        result_graph.add_edge(node_map[from_val], node_map[to_val], edge.weight)
        path_values.append(str(from_val))

    path_str = "-".join(path_values + [str(best_path[0].value)])

    return result_graph, total_weight, "Успех", path_str
