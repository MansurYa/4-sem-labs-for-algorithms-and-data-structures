from graph import Graph, Node
from typing import List, Dict, Tuple, Optional
import random
import time


class Ant:
    """Класс, представляющий муравья в муравьином алгоритме."""

    def __init__(self, start_node: Node, template: Optional[Dict[Node, Node]] = None):
        """
        Инициализирует муравья с начальной вершиной и опциональным шаблоном.

        :param start_node: Начальная вершина
        :param template: Словарь шаблона (вершина -> следующая вершина)
        """
        self.current_node = start_node
        self.path = [start_node]
        self.visited = {start_node}
        self.template = template

    def move(self, graph: Graph, alpha: float, beta: float) -> bool:
        """
        Выполняет один шаг муравья, выбирая следующую вершину.

        :param graph: Граф
        :param alpha: Влияние феромона
        :param beta: Влияние эвристики
        :return: True, если шаг успешен, False иначе
        """
        if len(self.visited) == len(graph.nodes):
            return False

        if self.template and self.current_node in self.template:
            next_node = self.template[self.current_node]
            if next_node not in self.visited:
                self.current_node = next_node
                self.path.append(next_node)
                self.visited.add(next_node)
                return True


        probabilities = []
        total = 0
        for edge in self.current_node.outgoing_edges:
            if edge.to_node not in self.visited:
                pheromone = getattr(edge, 'pheromone', 0.1)
                heuristic = 1 / edge.weight
                prob = (pheromone ** alpha) * (heuristic ** beta)
                probabilities.append((edge.to_node, prob))
                total += prob

        if not probabilities:
            return False

        probabilities = [(node, prob / total) for node, prob in probabilities]
        r = random.random()
        cumulative = 0
        for node, prob in probabilities:
            cumulative += prob
            if r <= cumulative:
                self.current_node = node
                self.path.append(node)
                self.visited.add(node)
                return True
        return False


def initialize_pheromones(graph: Graph, initial_pheromone: float):
    """Инициализирует феромоны на рёбрах графа."""
    for edge in graph.edges:
        edge.pheromone = initial_pheromone


def update_pheromones(graph: Graph, ants: List[Ant], rho: float, Q: float):
    """
    Обновляет феромоны с испарением и добавлением от муравьёв.

    :param graph: Граф
    :param ants: Список муравьёв
    :param rho: Коэффициент испарения
    :param Q: Константа для обновления феромона
    """
    # Испарение
    for edge in graph.edges:
        edge.pheromone *= (1 - rho)

    # Добавление феромонов
    for ant in ants:
        if len(ant.path) == len(graph.nodes) + 1 and ant.path[0] == ant.path[-1]:
            path_length = calculate_path_length(graph, ant.path)
            if path_length > 0 and path_length != float('inf'):
                delta = Q / path_length
                for i in range(len(ant.path) - 1):
                    from_node = ant.path[i]
                    to_node = ant.path[i + 1]
                    edge = next((e for e in from_node.outgoing_edges if e.to_node == to_node), None)
                    if edge:
                        edge.pheromone += delta


def calculate_path_length(graph: Graph, path: List[Node]) -> int:
    """
    Вычисляет длину пути.

    :param graph: Граф
    :param path: Список вершин пути
    :return: Длина пути или бесконечность, если путь невалиден
    """
    if len(path) != len(graph.nodes) + 1 or path[0] != path[-1]:
        return float('inf')
    total = 0
    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i + 1]
        edge = next((e for e in from_node.outgoing_edges if e.to_node == to_node), None)
        if not edge:
            return float('inf')
        total += edge.weight
    return total


def get_template(graph: Graph) -> Dict[Node, Node]:
    """
    Создаёт статический шаблон на основе минимальных весов рёбер.

    :param graph: Граф
    :return: Словарь шаблона
    """
    template = {}
    for node in graph.nodes:
        if node.outgoing_edges:
            min_weight = min(edge.weight for edge in node.outgoing_edges)
            min_edges = [edge for edge in node.outgoing_edges if edge.weight == min_weight]
            min_edge = random.choice(min_edges)
            template[node] = min_edge.to_node
    return template

def build_result_graph(graph: Graph, path: List[Node]) -> Graph:
    """
    Создаёт результирующий граф из найденного пути.

    :param graph: Исходный граф
    :param path: Список вершин пути
    :return: Новый граф с циклом
    """
    result_graph = Graph(directed=True)
    node_map = {node.value: result_graph.add_node(node.value) for node in graph.nodes}
    for node in result_graph.nodes:
        original_node = next(n for n in graph.nodes if n.value == node.value)
        node.x = original_node.x
        node.y = original_node.y

    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i + 1]
        edge = next((e for e in from_node.outgoing_edges if e.to_node == to_node), None)
        if edge:
            result_graph.add_edge(node_map[from_node.value], node_map[to_node.value], edge.weight)
    return result_graph


def ant_algorithm(
    graph: Graph,
    use_template: bool = False,
    num_ants: int = 10,
    num_iterations: int = 100,
    alpha: float = 1.0,
    beta: float = 2.0,
    rho: float = 0.1,
    Q: float = 100.0,
    initial_pheromone: float = 0.1
) -> Tuple[Graph, int, str, str]:
    """
    Реализует муравьиный алгоритм для поиска гамильтонова цикла с опциональной модификацией 'Шаблон'.

    :param graph: Входной граф
    :param use_template: Использовать ли модификацию с шаблоном
    :param num_ants: Количество муравьёв
    :param num_iterations: Количество итераций
    :param alpha: Влияние феромона
    :param beta: Влияние эвристики
    :param rho: Коэффициент испарения феромона
    :param Q: Константа для обновления феромона
    :param initial_pheromone: Начальный уровень феромона
    :return: (Граф с циклом, длина цикла, статус, строка пути)
    """
    start_time = time.time()

    if not graph.nodes or not graph.edges:
        return Graph(), 0, "Граф пуст", ""

    template = get_template(graph) if use_template else None

    initialize_pheromones(graph, initial_pheromone)
    best_path = None
    best_length = float('inf')

    for iteration in range(num_iterations):
        ants = [Ant(random.choice(graph.nodes), template) for _ in range(num_ants)]
        for ant in ants:
            while len(ant.visited) < len(graph.nodes):
                if not ant.move(graph, alpha, beta):
                    break
            if len(ant.visited) == len(graph.nodes) and ant.path[0] in ant.current_node.neighbors:
                ant.path.append(ant.path[0])
            if len(ant.path) == len(graph.nodes) + 1 and ant.path[0] == ant.path[-1]:
                path_length = calculate_path_length(graph, ant.path)
                if path_length < best_length:
                    best_length = path_length
                    best_path = ant.path.copy()
        update_pheromones(graph, ants, rho, Q)

    if best_path:
        result_graph = build_result_graph(graph, best_path)
        total_weight = calculate_path_length(graph, best_path)
        path_str = "-".join(str(node.value) for node in best_path)
        return result_graph, total_weight, "Успех", path_str
    return Graph(), 0, "Не удалось найти гамильтонов цикл", ""
