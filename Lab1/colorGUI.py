import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import math
import time
import platform  # Добавлен импорт для определения ОС
from nearest_neighbor_method import nearest_neighbor_method
from graph import Graph


class LabApp(tk.Tk):
    def __init__(self, graph):
        super().__init__()
        self.title("Метод ближайшего соседа")
        self.geometry("1250x700")

        self.columnconfigure(0, weight=1, minsize=200)
        self.columnconfigure(1, weight=2, minsize=700)
        self.columnconfigure(2, weight=1, minsize=350)
        self.rowconfigure(0, weight=1)

        self.graph = graph
        self.node_clicked = None
        self.answer_graph = None
        self.tree = None
        self.path_length_entry = None
        self.path_text = None
        self.output_canvas = None
        self.modification_var = tk.BooleanVar(value=False)
        self.time_entry = None  # Добавлено поле для времени

        self.create_widgets()
        self.draw_initial_graph()

    def create_widgets(self):
        """Создаёт все элементы интерфейса"""
        BG_COLOR = "#F0F0F0"
        ACCENT_1 = "#FF6B6B"
        ACCENT_2 = "#4ECDC4"
        DARK_TEXT = "#2D3436"
        LIGHT_TEXT = "#FFFFFF"
        PANEL_COLOR = "#FFEAA7"
        BTN_COLOR = "#55EFC4"
        CANVAS_BG = "#DFE6E9"

        self.configure(bg=BG_COLOR)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview.Heading", background=ACCENT_1, foreground=LIGHT_TEXT, font=('Helvetica', 10, 'bold'))
        style.configure("Treeview", background="#FFFFFF", fieldbackground="#FFFFFF", foreground=DARK_TEXT)
        style.map("Treeview", background=[('selected', ACCENT_2)])

        left_frame = tk.LabelFrame(
            self,
            padx=10,
            pady=10,
            text="Параметры",
            bg=PANEL_COLOR,
            fg=DARK_TEXT,
            font=('Helvetica', 12, 'bold')
        )
        left_frame.grid(row=0, column=0, sticky="nsew")

        modification_frame = tk.Frame(left_frame, bg=PANEL_COLOR)
        modification_frame.pack(fill="x", pady=5)

        tk.Label(
            modification_frame,
            text="Модификация:",
            bg=PANEL_COLOR,
            fg=DARK_TEXT,
            font=('Helvetica', 10)
        ).pack(side="left")

        modification_check = tk.Checkbutton(
            modification_frame,
            variable=self.modification_var,
            text="Все стартовые вершины",
            bg=PANEL_COLOR,
            fg=DARK_TEXT,
            selectcolor=ACCENT_2,
            activebackground=PANEL_COLOR,
            activeforeground=DARK_TEXT,
            font=('Helvetica', 10)
        )
        modification_check.pack(side="right")

        # Кнопка "Рассчитать"
        tk.Button(
            left_frame,
            text="Рассчитать",
            command=self.start_algorithm,
            bg=BTN_COLOR,
            fg=DARK_TEXT,
            activebackground=ACCENT_2,
            font=('Helvetica', 12, 'bold'),
            relief="flat"
        ).pack(fill="x", pady=10)

        # Поле времени выполнения (добавлено)
        tk.Label(
            left_frame,
            text="Время выполнения (сек)",
            bg=PANEL_COLOR,
            fg=DARK_TEXT,
            font=('Helvetica', 10)
        ).pack(anchor="w")

        self.time_entry = tk.Entry(
            left_frame,
            state="readonly",
            bg="#FFFFFF",
            fg=ACCENT_1,
            font=('Helvetica', 12, 'bold'),
            justify="center",
            readonlybackground="#FFFFFF"
        )
        self.time_entry.pack(fill="x", pady=2)
        self.time_entry.bind("<Button-3>", self.show_context_menu)
        self.time_entry.bind("<Control-c>", lambda e: self.copy_time())
        if platform.system() == "Darwin":
            self.time_entry.bind("<Command-c>", lambda e: self.copy_time())

        # Поле с путем
        tk.Label(
            left_frame,
            text="Полученный путь",
            bg=PANEL_COLOR,
            fg=DARK_TEXT,
            font=('Helvetica', 10)
        ).pack(anchor="w")

        self.path_text = tk.Text(
            left_frame,
            height=5,
            state="disabled",
            bg="#FFFFFF",
            fg=DARK_TEXT,
            font=('Courier New', 14)
        )
        self.path_text.pack(fill="both", pady=5, expand=True)

        # self.path_text = tk.Text(
        #     left_frame,
        #     height=5,
        #     state="disabled",
        #     bg="#FFFFFF",
        #     fg=DARK_TEXT,
        #     font=('Courier New', 14),
        #     wrap=tk.NONE,
        #     selectbackground=ACCENT_2,
        #     inactiveselectbackground=ACCENT_2,
        #     exportselection=1,
        #     highlightthickness=0,
        #     borderwidth=2,
        #     relief="solid"
        # )
        # self.path_text.pack(fill="both", pady=5, expand=True)
        # self.path_text.bind("<Button-3>", self.show_text_context_menu)
        # self.path_text.bind("<Control-c>", lambda e: self.copy_path_text())
        # self.path_text.bind("<Control-Insert>", lambda e: self.copy_path_text())
        # if platform.system() == "Darwin":
        #     self.path_text.bind("<Command-c>", lambda e: self.copy_path_text())

        # Поле длины пути
        tk.Label(
            left_frame,
            text="Длина пути",
            bg=PANEL_COLOR,
            fg=DARK_TEXT,
            font=('Helvetica', 10)
        ).pack(anchor="w")

        self.path_length_entry = tk.Entry(
            left_frame,
            state="readonly",
            bg="#FFFFFF",
            fg=ACCENT_1,
            font=('Helvetica', 12, 'bold'),
            justify="center"
        )
        self.path_length_entry.pack(fill="x", pady=2)

        # Центральная панель с графами
        center_frame = tk.Frame(self, bg=BG_COLOR)
        center_frame.grid(row=0, column=1, sticky="nsew")

        input_graph_frame = tk.LabelFrame(
            center_frame,
            text="Входной граф",
            padx=10,
            pady=10,
            bg=BG_COLOR,
            fg=DARK_TEXT,
            font=('Helvetica', 12, 'bold')
        )
        input_graph_frame.pack(fill="both", expand=True, side=tk.TOP)

        self.canvas = tk.Canvas(
            input_graph_frame,
            bg=CANVAS_BG,
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        clear_button = tk.Button(
            center_frame,
            text="Очистить входной граф",
            command=self.clear_graph,
            bg=ACCENT_1,
            fg=DARK_TEXT,
            activebackground="#FF8787",
            font=('Helvetica', 10),
            height=2,
        )
        clear_button.pack(fill="x", pady=5)

        output_graph_frame = tk.LabelFrame(
            center_frame,
            text="Выходной граф",
            padx=10,
            pady=10,
            bg=BG_COLOR,
            fg=DARK_TEXT,
            font=('Helvetica', 12, 'bold')
        )
        output_graph_frame.pack(fill="both", expand=True, side=tk.TOP)

        self.output_canvas = tk.Canvas(
            output_graph_frame,
            bg=CANVAS_BG,
            highlightthickness=0
        )
        self.output_canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Правая панель с таблицей ребер
        right_frame = tk.LabelFrame(
            self,
            text="Рёбра",
            padx=10,
            pady=10,
            bg=PANEL_COLOR,
            fg=DARK_TEXT,
            font=('Helvetica', 12, 'bold')
        )
        right_frame.grid(row=0, column=2, sticky="nsew")

        columns = ("from", "to", "weight", "visited")
        self.tree = ttk.Treeview(
            right_frame,
            columns=columns,
            show="headings",
            height=25,
        )

        for col in columns:
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

    # Добавленные методы для работы с буфером обмена
    def copy_to_clipboard(self, text):
        self.clipboard_clear()
        self.clipboard_append(text)
        if platform.system() == "Darwin":
            self.update()

    def show_context_menu(self, event):
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Копировать", command=self.copy_time)
        menu.tk.call("tk_popup", menu, event.x_root, event.y_root)

    def show_text_context_menu(self, event):
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Копировать", command=self.copy_path_text)
        menu.tk.call("tk_popup", menu, event.x_root, event.y_root)

    def copy_time(self):
        text = self.time_entry.get()
        self.copy_to_clipboard(text)

    def copy_path_text(self):
        try:
            text = self.path_text.get("sel.first", "sel.last")
        except tk.TclError:
            text = self.path_text.get("1.0", "end")
        self.copy_to_clipboard(text)

    # Остальные методы без изменений
    def on_canvas_click(self, event):
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
        for node in self.graph.nodes:
            if (abs(node.x - x) < 20) and (abs(node.y - y) < 20):
                return node
        return None

    def create_node(self, x, y):
        node = self.graph.add_node(len(self.graph.nodes))
        node.x = x
        node.y = y
        self.canvas.create_oval(x - 15, y - 15, x + 15, y + 15, fill="red")
        self.canvas.create_text(x, y, text=str(node.value), fill="white")
        print(f"Создана вершина {node.value} в ({x}, {y})")

    def create_edge(self, from_node, to_node):
        if from_node == to_node:
            print("Нелья создать петлю (ребро в саму себя).")
            return

        weight = simpledialog.askinteger("Вес ребра", f"Введите вес ребра {from_node.value} -> {to_node.value}")
        if weight is not None:
            self.graph.add_edge(from_node, to_node, weight)
            self.draw_edge(from_node, to_node, weight)
            self.update_edges_table()
            print(f"Создано ребро {from_node.value} -> {to_node.value} с весом {weight}")

    def draw_edge(self, from_node, to_node, weight):
        offset = 15

        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        angle = math.atan2(dy, dx)

        x1, y1 = from_node.x + offset * math.cos(angle), from_node.y + offset * math.sin(angle)
        x2, y2 = to_node.x - offset * math.cos(angle), to_node.y - offset * math.sin(angle)

        self.canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, width=3, fill="black")
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        self.canvas.create_text(mid_x, mid_y, text=str(weight), fill="black")

    def clear_graph(self):
        self.graph = Graph(directed=True)
        self.answer_graph = None
        self.canvas.delete("all")
        self.output_canvas.delete("all")
        self.update_edges_table()
        print("Граф очищен.")

    def start_algorithm(self):
        if self.modification_var.get():
            print("Режим с перебором всех стартовых вершин")
            self.graph.display()
            results = []
            start_time = time.time()

            for start_node in self.graph.nodes:
                result_graph, total_weight, result_mes, path_str = nearest_neighbor_method(
                    self.graph,
                    start_node=start_node
                )
                if "успешно" in result_mes:
                    results.append((result_graph, total_weight, result_mes, path_str))

            end_time = time.time()
            execution_time = end_time - start_time

            if not results:
                messagebox.showerror("Ошибка", "Не удалось найти ни одного гамильтонова цикла")
                return

            best_result = min(results, key=lambda x: x[1])
            self.answer_graph, answer_path_len, result_mes, path_str = best_result
        else:
            print("Обычный режим работы")
            self.graph.display()
            start_time = time.time()
            self.answer_graph, answer_path_len, result_mes, path_str = nearest_neighbor_method(self.graph)
            end_time = time.time()
            execution_time = end_time - start_time

        # Обновление времени выполнения
        self.time_entry.configure(state="normal")
        self.time_entry.delete(0, tk.END)
        self.time_entry.insert(0, f"{execution_time:.4f} сек")
        self.time_entry.configure(state="readonly")

        self.draw_graph(self.output_canvas, self.answer_graph)
        self.update_edges_table()
        messagebox.showinfo("Результат", result_mes)

        print(path_str)
        self.path_text.configure(state="normal")
        self.path_text.delete("1.0", tk.END)
        self.path_text.insert(tk.END, path_str)
        self.path_text.configure(state="disabled")

        self.path_length_entry.configure(state="normal")
        self.path_length_entry.delete(0, tk.END)
        self.path_length_entry.insert(0, str(answer_path_len))
        self.path_length_entry.configure(state="readonly")

    def draw_graph(self, canvas, graph):
        canvas.delete("all")

        for edge in graph.edges:
            self.draw_edge_on_canvas(canvas, edge.from_node, edge.to_node, edge.weight)

        for node in graph.nodes:
            self.draw_node_on_canvas(canvas, node)

    def draw_node_on_canvas(self, canvas, node):
        x, y = node.x, node.y
        canvas.create_oval(x - 15, y - 15, x + 15, y + 15, fill="blue" if canvas == self.output_canvas else "red")
        canvas.create_text(x, y, text=str(node.value), fill="white")

    def draw_edge_on_canvas(self, canvas, from_node, to_node, weight):
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
            is_used = (from_val, to_val, weight) in used_edges

            self.tree.insert("", "end", values=(
                from_val,
                to_val,
                weight,
                "Да" if is_used else ""
            ))

    def draw_initial_graph(self):
        for node in self.graph.nodes:
            self.canvas.create_oval(
                node.x - 15, node.y - 15,
                node.x + 15, node.y + 15,
                fill="red"
            )
            self.canvas.create_text(node.x, node.y, text=str(node.value), fill="white")

        for edge in self.graph.edges:
            self.draw_edge(edge.from_node, edge.to_node, edge.weight)


if __name__ == "__main__":
    graph = Graph(directed=True)
    app = LabApp(graph)
    app.mainloop()
