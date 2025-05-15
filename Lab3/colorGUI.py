import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import math
import time
import platform
from graph import Graph, Edge
from ant_algorithm import ant_algorithm


class LabApp(tk.Tk):
    def __init__(self, graph):
        super().__init__()
        self.title("Муравьиный алгоритм")
        self.geometry("1250x700")

        self.columnconfigure(0, weight=1, minsize=200)
        self.columnconfigure(1, weight=2, minsize=700)
        self.columnconfigure(2, weight=1, minsize=400)  # Увеличен minsize до 400
        self.rowconfigure(0, weight=1)

        self.graph = graph
        self.node_clicked = None
        self.answer_graph = None
        self.tree = None
        self.path_length_entry = None
        self.path_text = None
        self.output_canvas = None
        self.time_entry = None

        self._setup_styles()
        self.create_widgets()
        self.draw_initial_graph()

    def _setup_styles(self):
        """Настройка стилей интерфейса"""
        self.BG_COLOR = "#F0F0F0"
        self.ACCENT_1 = "#FF6B6B"
        self.ACCENT_2 = "#4ECDC4"
        self.DARK_TEXT = "#2D3436"
        self.LIGHT_TEXT = "#FFFFFF"
        self.PANEL_COLOR = "#FFEAA7"
        self.BTN_COLOR = "#55EFC4"
        self.CANVAS_BG = "#DFE6E9"

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview.Heading",
                       background=self.ACCENT_1,
                       foreground=self.LIGHT_TEXT,
                       font=('Helvetica', 10, 'bold'))
        style.configure("Treeview",
                       background="#FFFFFF",
                       fieldbackground="#FFFFFF",
                       foreground=self.DARK_TEXT)
        style.map("Treeview", background=[('selected', self.ACCENT_2)])

    def create_widgets(self):
        """Создание всех элементов интерфейса"""
        left_frame = tk.LabelFrame(
            self,
            padx=10,
            pady=10,
            text="Параметры алгоритма",
            bg=self.PANEL_COLOR,
            fg=self.DARK_TEXT,
            font=('Helvetica', 12, 'bold')
        )
        left_frame.grid(row=0, column=0, sticky="nsew")

        # Параметры алгоритма
        param_frame = tk.Frame(left_frame, bg=self.PANEL_COLOR)
        param_frame.pack(fill="x", pady=5)

        params = [
            ("Количество муравьёв:", "num_ants", "100"),
            ("Количество итераций:", "num_iterations", "300"),
            ("Alpha (феромон):", "alpha", "1.0"),
            ("Beta (эвристика):", "beta", "2.0"),
            ("Rho (испарение):", "rho", "0.1"),
            ("Q (константа):", "Q", "100.0"),
            ("Начальный феромон:", "initial_pheromone", "0.1"),
        ]

        for i, (label_text, var_name, default) in enumerate(params):
            tk.Label(
                param_frame,
                text=label_text,
                bg=self.PANEL_COLOR,
                fg=self.DARK_TEXT,
                font=('Helvetica', 10)
            ).grid(row=i, column=0, sticky="w", padx=5, pady=2)

            entry = tk.Entry(param_frame, width=15)
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=5, pady=2)
            setattr(self, f"{var_name}_entry", entry)

        self.use_template_var = tk.BooleanVar()
        tk.Checkbutton(
            param_frame,
            text="Использовать шаблон",
            variable=self.use_template_var,
            bg=self.PANEL_COLOR,
            font=('Helvetica', 10)
        ).grid(row=len(params), columnspan=2, pady=5)

        # Кнопка запуска
        tk.Button(
            left_frame,
            text="Запустить алгоритм",
            command=self.start_algorithm,
            bg=self.BTN_COLOR,
            fg=self.DARK_TEXT,
            activebackground=self.ACCENT_2,
            font=('Helvetica', 12, 'bold')
        ).pack(fill="x", pady=10)

        # Поле времени выполнения
        tk.Label(
            left_frame,
            text="Время выполнения (сек):",
            bg=self.PANEL_COLOR,
            fg=self.DARK_TEXT,
            font=('Helvetica', 10)
        ).pack(anchor="w")

        self.time_entry = tk.Entry(
            left_frame,
            state="readonly",
            bg="#FFFFFF",
            fg=self.ACCENT_1,
            font=('Helvetica', 12, 'bold'),
            justify="center",
            readonlybackground="#FFFFFF"
        )
        self.time_entry.pack(fill="x", pady=2)
        self.time_entry.bind("<Button-3>", self.show_context_menu)
        self.time_entry.bind("<Control-c>", lambda e: self.copy_time())
        if platform.system() == "Darwin":
            self.time_entry.bind("<Command-c>", lambda e: self.copy_time())

        # Результаты
        tk.Label(
            left_frame,
            text="Полученный путь:",
            bg=self.PANEL_COLOR,
            fg=self.DARK_TEXT,
            font=('Helvetica', 10)
        ).pack(anchor="w")

        self.path_text = tk.Text(
            left_frame,
            height=5,
            state="disabled",
            bg="#FFFFFF",
            fg="#2D3436",
            font=('Courier New', 14)
        )
        self.path_text.pack(fill="both", pady=5, expand=True)

        tk.Label(
            left_frame,
            text="Длина пути:",
            bg=self.PANEL_COLOR,
            fg=self.DARK_TEXT,
            font=('Helvetica', 10)
        ).pack(anchor="w")

        self.path_length_entry = tk.Entry(
            left_frame,
            state="readonly",
            bg="#FFFFFF",
            fg=self.ACCENT_1,
            font=('Helvetica', 12, 'bold'),
            justify="center",
            readonlybackground="#FFFFFF"
        )
        self.path_length_entry.pack(fill="x", pady=2)
        self.path_length_entry.bind("<Button-3>", self.show_context_menu)
        self.path_length_entry.bind("<Control-c>", lambda e: self.copy_path_length())
        if platform.system() == "Darwin":
            self.path_length_entry.bind("<Command-c>", lambda e: self.copy_path_length())

        # Центральная панель с графами
        center_frame = tk.Frame(self, bg=self.BG_COLOR)
        center_frame.grid(row=0, column=1, sticky="nsew")

        # Входной граф
        input_graph_frame = tk.LabelFrame(
            center_frame,
            text="Исходный граф",
            padx=10,
            pady=10,
            bg=self.BG_COLOR,
            fg=self.DARK_TEXT,
            font=('Helvetica', 12, 'bold')
        )
        input_graph_frame.pack(fill="both", expand=True, side=tk.TOP)

        self.input_canvas = tk.Canvas(
            input_graph_frame,
            bg=self.CANVAS_BG,
            highlightthickness=0
        )
        self.input_canvas.pack(fill="both", expand=True)
        self.input_canvas.bind("<Button-1>", self.on_canvas_click)

        # Кнопка очистки
        tk.Button(
            center_frame,
            text="Очистить граф",
            command=self.clear_graph,
            bg=self.ACCENT_1,
            fg=self.DARK_TEXT,
            activebackground="#FF8787",
            font=('Helvetica', 10),
            height=2
        ).pack(fill="x", pady=5)

        # Выходной граф
        output_graph_frame = tk.LabelFrame(
            center_frame,
            text="Результат",
            padx=10,
            pady=10,
            bg=self.BG_COLOR,
            fg=self.DARK_TEXT,
            font=('Helvetica', 12, 'bold')
        )
        output_graph_frame.pack(fill="both", expand=True, side=tk.TOP)

        self.output_canvas = tk.Canvas(
            output_graph_frame,
            bg=self.CANVAS_BG,
            highlightthickness=0
        )
        self.output_canvas.pack(fill="both", expand=True)

        # Правая панель с таблицей ребер
        right_frame = tk.LabelFrame(
            self,
            text="Рёбра графа",
            padx=10,
            pady=10,
            bg=self.PANEL_COLOR,
            fg=self.DARK_TEXT,
            font=('Helvetica', 12, 'bold')
        )
        right_frame.grid(row=0, column=2, sticky="nsew")

        self.tree = ttk.Treeview(
            right_frame,
            columns=("from", "to", "weight", "pheromone", "used"),  # Добавлена колонка pheromone
            show="headings",
            height=25
        )

        for col in ("from", "to", "weight", "pheromone", "used"):
            self.tree.heading(col, text=col.capitalize())
            self.tree.column(col, width=80, anchor="center")

        scrollbar = ttk.Scrollbar(
            right_frame,
            orient="vertical",
            command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    # Методы для работы с буфером обмена
    def copy_to_clipboard(self, text):
        self.clipboard_clear()
        self.clipboard_append(text)
        if platform.system() == "Darwin":
            self.update()

    def show_context_menu(self, event):
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Копировать", command=self.copy_time)
        menu.tk.call("tk_popup", menu, event.x_root, event.y_root)

    def copy_time(self):
        text = self.time_entry.get()
        self.copy_to_clipboard(text)

    def copy_path_length(self):
        text = self.path_length_entry.get()
        self.copy_to_clipboard(text)

    def on_canvas_click(self, event):
        """Обработчик клика по холсту"""
        x, y = event.x, event.y
        node = self._find_node_at(x, y)

        if node:
            if self.node_clicked:
                self._create_edge(self.node_clicked, node)
                self.node_clicked = None
            else:
                self.node_clicked = node
        else:
            if not self.node_clicked:
                self._create_node(x, y)

    def _find_node_at(self, x, y):
        """Поиск узла по координатам"""
        for node in self.graph.nodes:
            if abs(node.x - x) < 15 and abs(node.y - y) < 15:
                return node
        return None

    def _create_node(self, x, y):
        """Создание нового узла"""
        node = self.graph.add_node(len(self.graph.nodes))
        node.x = x
        node.y = y
        self._draw_node(node, self.input_canvas, "red")
        self.update_edges_table()

    def _create_edge(self, from_node, to_node):
        """Создание нового ребра"""
        if from_node == to_node:
            return

        weight = simpledialog.askinteger(
            "Вес ребра",
            f"Введите вес для {from_node.value} → {to_node.value}:"
        )

        if weight:
            self.graph.add_edge(from_node, to_node, weight)
            self._draw_edge(Edge(from_node, to_node, weight), self.input_canvas)
            self.update_edges_table()

    def _draw_node(self, node, canvas, color):
        """Отрисовка узла на холсте"""
        x, y = node.x, node.y
        canvas.create_oval(x-15, y-15, x+15, y+15, fill=color, outline="")
        canvas.create_text(x, y, text=str(node.value), fill="white")

    def _draw_edge(self, edge, canvas):
        """Отрисовка ребра на холсте"""
        from_node = edge.from_node
        to_node = edge.to_node

        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        angle = math.atan2(dy, dx)

        offset = 15
        x1 = from_node.x + offset * math.cos(angle)
        y1 = from_node.y + offset * math.sin(angle)
        x2 = to_node.x - offset * math.cos(angle)
        y2 = to_node.y - offset * math.sin(angle)

        canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, width=2)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        canvas.create_text(mid_x, mid_y, text=str(edge.weight))

    def draw_initial_graph(self):
        """Отрисовка исходного графа при запуске"""
        for node in self.graph.nodes:
            self._draw_node(node, self.input_canvas, "red")

        for edge in self.graph.edges:
            self._draw_edge(edge, self.input_canvas)

    def clear_graph(self):
        """Очистка графа"""
        self.graph = Graph(directed=True)
        self.input_canvas.delete("all")
        self.output_canvas.delete("all")
        self.update_edges_table()

    def start_algorithm(self):
        """Запуск муравьиного алгоритма"""
        print("start")
        try:
            params = {
                'use_template': self.use_template_var.get(),
                'num_ants': int(self.num_ants_entry.get()),
                'num_iterations': int(self.num_iterations_entry.get()),
                'alpha': float(self.alpha_entry.get()),
                'beta': float(self.beta_entry.get()),
                'rho': float(self.rho_entry.get()),
                'Q': float(self.Q_entry.get()),
                'initial_pheromone': float(self.initial_pheromone_entry.get())
            }

            if params['rho'] >= 1 or params['rho'] <= 0:
                raise ValueError("Rho (испарение) должно быть между 0 и 1")

        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректные параметры: {str(e)}")
            return

        try:
            start_time = time.time()
            result = ant_algorithm(
                self.graph,
                use_template=params['use_template'],
                num_ants=params['num_ants'],
                num_iterations=params['num_iterations'],
                alpha=params['alpha'],
                beta=params['beta'],
                rho=params['rho'],
                Q=params['Q'],
                initial_pheromone=params['initial_pheromone']
            )
            end_time = time.time()
            execution_time = end_time - start_time

            self.time_entry.configure(state="normal")
            self.time_entry.delete(0, tk.END)
            self.time_entry.insert(0, f"{execution_time:.4f}")
            self.time_entry.configure(state="readonly")

            self.answer_graph, length, status, path = result
            self._update_results(length, status, path)
            self._draw_result_graph()
            self.update_edges_table()
            print(path)

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _update_results(self, length, status, path):
        """Обновление результатов в интерфейсе"""
        self.path_text.configure(state="normal")
        self.path_text.delete("1.0", tk.END)
        self.path_text.insert(tk.END, path)
        self.path_text.configure(state="disabled")

        self.path_length_entry.configure(state="normal")
        self.path_length_entry.delete(0, tk.END)
        self.path_length_entry.insert(0, str(length))
        self.path_length_entry.configure(state="readonly")

        messagebox.showinfo("Результат", status)

    def _draw_result_graph(self):
        """Отрисовка результирующего графа"""
        self.output_canvas.delete("all")

        for node in self.answer_graph.nodes:
            self._draw_node(node, self.output_canvas, "blue")

        for edge in self.answer_graph.edges:
            self._draw_edge(edge, self.output_canvas)

    def update_edges_table(self):
        """Обновление таблицы ребер"""
        for item in self.tree.get_children():
            self.tree.delete(item)

        used_edges = set()
        if self.answer_graph:
            for edge in self.answer_graph.edges:
                key = (edge.from_node.value, edge.to_node.value, edge.weight)
                used_edges.add(key)

        sorted_edges = sorted(
            self.graph.edges,
            key=lambda edge: (edge.from_node.value, edge.to_node.value)
        )

        for edge in sorted_edges:
            from_val = edge.from_node.value
            to_val = edge.to_node.value
            weight = edge.weight
            pheromone = getattr(edge, 'pheromone', 0.0)  # Получаем феромон, по умолчанию 0.0
            is_used = (from_val, to_val, weight) in used_edges

            self.tree.insert("", "end", values=(
                from_val,
                to_val,
                weight,
                f"{pheromone:.4f}",  # Отображаем феромон с 4 знаками после запятой
                "✓" if is_used else ""
            ))


if __name__ == "__main__":
    initial_graph = Graph(directed=True)
    app = LabApp(initial_graph)
    app.mainloop()
