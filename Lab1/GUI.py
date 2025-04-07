import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import math
from nearest_neighbor_method import nearest_neighbor_method
from graph import Graph


class LabApp(tk.Tk):
    def __init__(self, graph):
        super().__init__()
        self.title("Метод ближайшего соседа")
        self.geometry("1200x600")

        self.columnconfigure(0, weight=1, minsize=300)
        self.columnconfigure(1, weight=2, minsize=600)
        self.columnconfigure(2, weight=1, minsize=300)
        self.rowconfigure(0, weight=1)

        self.graph = graph
        self.node_clicked = None
        self.answer_graph = None
        self.tree = None
        self.path_length_entry = None
        self.path_text = None
        self.output_canvas = None
        self.modification_var = tk.BooleanVar(value=False)  # Переменная для хранения состояния модификации

        self.create_widgets()
        self.draw_initial_graph()

    def create_widgets(self):
        """Создаёт все элементы интерфейса"""

        left_frame = tk.LabelFrame(self, padx=10, pady=10, text="Параметры")
        left_frame.grid(row=0, column=0, sticky="nsew")

        # Добавляем переключатель модификации
        modification_frame = tk.Frame(left_frame)
        modification_frame.pack(fill="x", pady=5)
        tk.Label(modification_frame, text="Модификация:").pack(side="left")
        modification_check = tk.Checkbutton(
            modification_frame,
            variable=self.modification_var,
            text="Все стартовые вершины",
            command=lambda: None
        )
        modification_check.pack(side="right")
        tk.Label(modification_frame, text="Перебор всех стартовых вершин").pack(side="left")

        tk.Button(left_frame, text="Рассчитать", command=self.start_algorithm).pack(fill="x", pady=5)

        tk.Label(left_frame, text="Полученный путь").pack(anchor="w")
        self.path_text = tk.Text(left_frame, height=5, state="disabled")
        self.path_text.pack(fill="both", pady=5, expand=True)

        tk.Label(left_frame, text="Длина пути").pack(anchor="w")
        self.path_length_entry = tk.Entry(left_frame, state="readonly")
        self.path_length_entry.pack(fill="x", pady=2)

        center_frame = tk.Frame(self)
        center_frame.grid(row=0, column=1, sticky="nsew")

        input_graph_frame = tk.LabelFrame(center_frame, text="Входной граф", padx=10, pady=10)
        input_graph_frame.pack(fill="both", expand=True, side=tk.TOP)

        self.canvas = tk.Canvas(input_graph_frame, bg="white")
        self.canvas.pack(fill="both", expand=True)

        clear_button = tk.Button(center_frame, text="Очистить входной граф", command=self.clear_graph)
        clear_button.pack(fill="x", pady=5)

        output_graph_frame = tk.LabelFrame(center_frame, text="Выходной граф", padx=10, pady=10)
        output_graph_frame.pack(fill="both", expand=True, side=tk.TOP)

        self.output_canvas = tk.Canvas(output_graph_frame, bg="white")
        self.output_canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Button-1>", self.on_canvas_click)

        right_frame = tk.LabelFrame(self, text="Рёбра", padx=10, pady=10)
        right_frame.grid(row=0, column=2, sticky="nsew")

        columns = ("from", "to", "weight", "visited")
        self.tree = ttk.Treeview(right_frame, columns=columns, show="headings", height=25)

        self.tree.heading("from", text="От")
        self.tree.heading("to", text="К")
        self.tree.heading("weight", text="Длина")
        self.tree.heading("visited", text="Посещено")

        self.tree.column("from", width=50, anchor="center")
        self.tree.column("to", width=50, anchor="center")
        self.tree.column("weight", width=70, anchor="center")
        self.tree.column("visited", width=80, anchor="center")

        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def on_canvas_click(self, event):
        """Обрабатывает клик на холсте"""
        x, y = event.x, event.y
        node = self.get_node_at(x, y)

        if node is None:
            if self.node_clicked is None:
                self.create_node(x, y)
            else:
                print("Выбор второй вершины отменён.")
                self.node_clicked = None
        else:
            if self.node_clicked is None:
                print(f"Выбрана вершина {node.value}. Ожидание второй вершины.")
                self.node_clicked = node
            else:
                self.create_edge(self.node_clicked, node)
                self.node_clicked = None

    def get_node_at(self, x, y):
        """Проверяет, есть ли вершина в области клика"""
        for node in self.graph.nodes:
            if (abs(node.x - x) < 20) and (abs(node.y - y) < 20):
                return node
        return None

    def create_node(self, x, y):
        """Создаёт вершину в указанной позиции"""
        node = self.graph.add_node(len(self.graph.nodes))
        node.x = x
        node.y = y
        self.canvas.create_oval(x - 15, y - 15, x + 15, y + 15, fill="red")
        self.canvas.create_text(x, y, text=str(node.value), fill="white")
        print(f"Создана вершина {node.value} в ({x}, {y})")

    def create_edge(self, from_node, to_node):
        """Создаёт ребро между вершинами"""
        if from_node == to_node:
            print("Нельзя создать петлю (ребро в саму себя).")
            return

        weight = simpledialog.askinteger("Вес ребра", f"Введите вес ребра {from_node.value} -> {to_node.value}")
        if weight is not None:
            self.graph.add_edge(from_node, to_node, weight)
            self.draw_edge(from_node, to_node, weight)
            self.update_edges_table()
            print(f"Создано ребро {from_node.value} -> {to_node.value} с весом {weight}")

    def draw_edge(self, from_node, to_node, weight):
        """Рисует ребро между двумя вершинами с учётом стрелки"""
        offset = 15

        # Вычисляем угол между вершинами
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        angle = math.atan2(dy, dx)

        # Вычисляем точки для стрелки, чтобы она касалась окружности
        x1, y1 = from_node.x + offset * math.cos(angle), from_node.y + offset * math.sin(angle)
        x2, y2 = to_node.x - offset * math.cos(angle), to_node.y - offset * math.sin(angle)

        self.canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, width=3, fill="black")
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        self.canvas.create_text(mid_x, mid_y, text=str(weight), fill="black")

    def clear_graph(self):
        """Очищает весь граф и холсты"""
        self.graph = Graph(directed=True)
        self.answer_graph = None
        self.canvas.delete("all")
        self.output_canvas.delete("all")
        self.update_edges_table()  # Очищаем таблицу
        print("Граф очищен.")

    def start_algorithm(self):
        """Запуск алгоритма и обновление интерфейса"""
        if self.modification_var.get():
            print("Режим с перебором всех стартовых вершин")
            self.graph.display()
            results = []
            for start_node in self.graph.nodes:
                # print("start_node", start_node)
                result_graph, total_weight, result_mes, path_str = nearest_neighbor_method(
                    self.graph,
                    start_node=start_node
                )
                # print(path_str)
                if "успешно" in result_mes:
                    results.append((result_graph, total_weight, result_mes, path_str))

            if not results:
                messagebox.showerror("Ошибка", "Не удалось найти ни одного гамильтонова цикла")
                return

            # Выбираем результат с минимальной длиной
            best_result = min(results, key=lambda x: x[1])
            self.answer_graph, answer_path_len, result_mes, path_str = best_result
        else:
            print("Обычный режим работы")
            self.graph.display()
            self.answer_graph, answer_path_len, result_mes, path_str = nearest_neighbor_method(self.graph)

        self.draw_graph(self.output_canvas, self.answer_graph)

        self.update_edges_table()

        messagebox.showinfo("Результат", result_mes)

        self.path_text.configure(state="normal")
        self.path_text.delete("1.0", tk.END)
        self.path_text.insert(tk.END, path_str)
        self.path_text.configure(state="disabled")

        self.path_length_entry.configure(state="normal")
        self.path_length_entry.delete(0, tk.END)
        self.path_length_entry.insert(0, str(answer_path_len))
        self.path_length_entry.configure(state="readonly")

    def draw_graph(self, canvas, graph):
        """Отрисовывает граф на указанном холсте"""
        canvas.delete("all")  # Очищаем предыдущий рисунок

        for edge in graph.edges:
            self.draw_edge_on_canvas(canvas, edge.from_node, edge.to_node, edge.weight)

        for node in graph.nodes:
            self.draw_node_on_canvas(canvas, node)

    def draw_node_on_canvas(self, canvas, node):
        """Рисует узел на указанном холсте"""
        x, y = node.x, node.y
        canvas.create_oval(x - 15, y - 15, x + 15, y + 15, fill="blue" if canvas == self.output_canvas else "red")
        canvas.create_text(x, y, text=str(node.value), fill="white")

    def draw_edge_on_canvas(self, canvas, from_node, to_node, weight):
        """Рисует ребро на указанном холсте"""
        offset = 15
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        angle = math.atan2(dy, dx)

        x1 = from_node.x + offset * math.cos(angle)
        y1 = from_node.y + offset * math.sin(angle)
        x2 = to_node.x - offset * math.cos(angle)
        y2 = to_node.y - offset * math.sin(angle)

        canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, width=3,
                           fill="darkgray" if canvas == self.output_canvas else "black")
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        canvas.create_text(mid_x, mid_y, text=str(weight), fill="black")

    def update_edges_table(self):
        """Обновляет таблицу рёбер с сортировкой по 'От' и 'К'"""
        for item in self.tree.get_children():
            self.tree.delete(item)

        used_edges = set()
        if self.answer_graph:
            for edge in self.answer_graph.edges:
                key = (edge.from_node.value, edge.to_node.value, edge.weight)
                used_edges.add(key)

        # Сортируем рёбра сначала по "От", затем по "К"
        sorted_edges = sorted(
            self.graph.edges,
            key=lambda edge: (edge.from_node.value, edge.to_node.value)
        )

        for edge in sorted_edges:
            from_val = edge.from_node.value
            to_val = edge.to_node.value
            weight = edge.weight
            is_used = (from_val, to_val, weight) in used_edges

            self.tree.insert("", "end", values=(
                from_val,
                to_val,
                weight,
                "Да" if is_used else ""
            ))

    def draw_initial_graph(self):
        """Отрисовывает существующие узлы и рёбра при запуске"""
        for node in self.graph.nodes:
            self.canvas.create_oval(
                node.x - 15, node.y - 15,
                node.x + 15, node.y + 15,
                fill="red"
            )
            self.canvas.create_text(node.x, node.y, text=str(node.value), fill="white")

        for edge in self.graph.edges:
            self.draw_edge(edge.from_node, edge.to_node, edge.weight)
