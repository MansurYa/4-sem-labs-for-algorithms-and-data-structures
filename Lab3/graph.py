class Node:
    """Класс для представления вершины графа"""
    def __init__(self, value):
        self.value = value
        self.neighbors = []
        self.outgoing_edges = []
        self.x = None
        self.y = None
        self.visited = False

    def add_neighbor(self, node):
        """Добавить соседа"""
        self.neighbors.append(node)

    def sorted_edges(self):
        """Возвращает отсортированный список исходящих рёбер по весу"""
        return sorted(self.outgoing_edges, key=lambda edge: edge.weight)

    def __repr__(self):
        return f"Node(value={self.value}, (x,y)=({self.x},{self.y}))"


class Edge:
    """Класс для представления ребра"""
    def __init__(self, from_node: Node, to_node: Node, weight: int = 1):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight

    def __repr__(self):
        return f"Edge({self.from_node.value} -> {self.to_node.value}, weight={self.weight})"


class Graph:
    """Класс для представления графа"""
    def __init__(self, directed=True):
        self.nodes = []
        self.edges = []
        self.directed = directed

    def add_node(self, value):
        """Добавить вершину в граф"""
        node = Node(value)
        self.nodes.append(node)
        return node

    def add_edge(self, from_node, to_node, weight=1):
        """Добавить ребро в граф"""
        forward_edge = Edge(from_node, to_node, weight)
        self.edges.append(forward_edge)
        from_node.add_neighbor(to_node)
        from_node.outgoing_edges.append(forward_edge)
        if not self.directed:
            reverse_edge = Edge(to_node, from_node, weight)
            self.edges.append(reverse_edge)
            to_node.add_neighbor(from_node)
            to_node.outgoing_edges.append(reverse_edge)

    def display(self):
        """Выводит все вершины и рёбра"""
        print("Nodes:")
        for node in self.nodes:
            print(node)
        print("Edges:")
        for edge in self.edges:
            print(edge)
