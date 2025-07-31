"""
SINGLE PLOT CLUSTERING VISUALIZATION GUI
GUI для визуализации результатов кластеризации с одним большим 2D графиком
Загружает результаты из CSV файла, созданного ExtendedClusteringProcessor
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import json
from pathlib import Path
import logging
from typing import Optional, List, Dict, Any, Tuple
import re

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SinglePlotClusteringApp:
    """
    Приложение для визуализации результатов кластеризации с одним большим графиком.

    Особенности:
    - Загрузка результатов из CSV файла (K=100→2)
    - Интерактивный выбор K для отображения
    - Большой график с высоким качеством визуализации
    - Подробная статистика по кластерам
    - Навигация и масштабирование графика
    """

    def __init__(self, root: tk.Tk):
        """
        Инициализация приложения.

        :param root: Корневое окно tkinter
        """
        self.root = root
        self.root.title("Визуализация кластеризации - Одиночный график")
        self.root.geometry("1200x800")

        # === ДАННЫЕ ПРИЛОЖЕНИЯ ===
        self.clustering_data = None       # DataFrame с результатами кластеризации
        self.available_k_values = []     # Список доступных значений K
        self.feature_columns = []        # Названия столбцов с признаками
        self.metadata = {}               # Метаданные процесса кластеризации

        # === GUI ПЕРЕМЕННЫЕ ===
        self.csv_file_var = tk.StringVar()
        self.selected_k_var = tk.StringVar(value="10")
        self.color_scheme_var = tk.StringVar(value="tab10")
        self.point_size_var = tk.IntVar(value=50)
        self.transparency_var = tk.DoubleVar(value=0.7)

        # === СОЗДАНИЕ ИНТЕРФЕЙСА ===
        self._create_gui_layout()

        # === АВТОЗАГРУЗКА ===
        self._auto_load_latest_results()

    def _create_gui_layout(self):
        """Создание макета графического интерфайса."""

        # === ГЛАВНОЕ РАЗДЕЛЕНИЕ ===
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # === ЛЕВАЯ ПАНЕЛЬ УПРАВЛЕНИЯ ===
        left_frame = ttk.Frame(main_paned, width=300)
        main_paned.add(left_frame, weight=0)

        # === ПРАВАЯ ПАНЕЛЬ ВИЗУАЛИЗАЦИИ ===
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        # Создание секций левой панели
        self._create_file_section(left_frame)
        self._create_cluster_selection_section(left_frame)
        self._create_visualization_settings_section(left_frame)
        self._create_statistics_section(left_frame)
        self._create_controls_section(left_frame)

        # Создание правой панели визуализации
        self._create_visualization_panel(right_frame)

    def _create_file_section(self, parent):
        """Создание секции работы с файлами."""
        file_frame = ttk.LabelFrame(parent, text="📁 Файл результатов", padding=10)
        file_frame.pack(fill=tk.X, pady=5)

        # Поле выбора файла
        file_entry = ttk.Entry(file_frame, textvariable=self.csv_file_var, state="readonly")
        file_entry.pack(fill=tk.X, pady=2)

        # Кнопки управления файлами
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Выбрать файл...",
                  command=self._browse_csv_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Обновить",
                  command=self._reload_current_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Авто",
                  command=self._auto_load_latest_results).pack(side=tk.LEFT, padx=2)

    def _create_cluster_selection_section(self, parent):
        """Создание секции выбора количества кластеров."""
        cluster_frame = ttk.LabelFrame(parent, text="🎯 Выбор кластеров", padding=10)
        cluster_frame.pack(fill=tk.X, pady=5)

        # Выбор K
        k_frame = ttk.Frame(cluster_frame)
        k_frame.pack(fill=tk.X, pady=2)

        ttk.Label(k_frame, text="Количество кластеров (K):").pack(anchor=tk.W)

        self.k_combobox = ttk.Combobox(k_frame, textvariable=self.selected_k_var,
                                      state="readonly", width=10)
        self.k_combobox.pack(fill=tk.X, pady=2)
        self.k_combobox.bind('<<ComboboxSelected>>', self._on_k_changed)

        # Информация о выбранном K
        self.k_info_label = ttk.Label(cluster_frame, text="Выберите файл с результатами",
                                     foreground="gray")
        self.k_info_label.pack(anchor=tk.W, pady=5)

    def _create_visualization_settings_section(self, parent):
        """Создание секции настроек визуализации."""
        viz_frame = ttk.LabelFrame(parent, text="🎨 Настройки визуализации", padding=10)
        viz_frame.pack(fill=tk.X, pady=5)

        # Цветовая схема
        ttk.Label(viz_frame, text="Цветовая схема:").pack(anchor=tk.W)
        color_combo = ttk.Combobox(viz_frame, textvariable=self.color_scheme_var,
                                  values=["tab10", "Set1", "Pastel1", "Dark2", "rainbow", "viridis", "plasma"],
                                  state="readonly")
        color_combo.pack(fill=tk.X, pady=2)
        color_combo.bind('<<ComboboxSelected>>', self._on_visualization_changed)

        # Размер точек
        ttk.Label(viz_frame, text="Размер точек:").pack(anchor=tk.W, pady=(10, 0))
        size_scale = ttk.Scale(viz_frame, from_=10, to=200, variable=self.point_size_var,
                              orient=tk.HORIZONTAL, command=self._on_visualization_changed)
        size_scale.pack(fill=tk.X, pady=2)

        self.size_label = ttk.Label(viz_frame, text=f"Размер: {self.point_size_var.get()}")
        self.size_label.pack(anchor=tk.W)

        # Прозрачность
        ttk.Label(viz_frame, text="Прозрачность:").pack(anchor=tk.W, pady=(10, 0))
        alpha_scale = ttk.Scale(viz_frame, from_=0.1, to=1.0, variable=self.transparency_var,
                               orient=tk.HORIZONTAL, command=self._on_visualization_changed)
        alpha_scale.pack(fill=tk.X, pady=2)

        self.alpha_label = ttk.Label(viz_frame, text=f"Прозрачность: {self.transparency_var.get():.1f}")
        self.alpha_label.pack(anchor=tk.W)

    def _create_statistics_section(self, parent):
        """Создание секции статистики."""
        stats_frame = ttk.LabelFrame(parent, text="📊 Статистика", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Текстовое поле для статистики
        self.stats_text = tk.Text(stats_frame, wrap=tk.WORD, height=10, font=('Courier', 9))
        stats_scrollbar = ttk.Scrollbar(stats_frame, command=self.stats_text.yview)
        self.stats_text.config(yscrollcommand=stats_scrollbar.set)

        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Изначально показываем инструкцию
        self._show_initial_instructions()

    def _create_controls_section(self, parent):
        """Создание секции кнопок управления."""
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, pady=5)

        # Основные кнопки
        btn_frame1 = ttk.Frame(controls_frame)
        btn_frame1.pack(fill=tk.X, pady=2)

        ttk.Button(btn_frame1, text="🔄 Обновить график",
                  command=self._update_plot).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        btn_frame2 = ttk.Frame(controls_frame)
        btn_frame2.pack(fill=tk.X, pady=2)

        ttk.Button(btn_frame2, text="💾 Сохранить график",
                  command=self._save_plot).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        btn_frame3 = ttk.Frame(controls_frame)
        btn_frame3.pack(fill=tk.X, pady=2)

        ttk.Button(btn_frame3, text="📈 Экспорт данных",
                  command=self._export_current_data).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

    def _create_visualization_panel(self, parent):
        """Создание панели визуализации."""
        viz_frame = ttk.LabelFrame(parent, text="📊 Визуализация кластеризации", padding=5)
        viz_frame.pack(fill=tk.BOTH, expand=True)

        # Создание matplotlib фигуры
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Canvas для matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Панель инструментов навигации
        toolbar_frame = ttk.Frame(viz_frame)
        toolbar_frame.pack(fill=tk.X)

        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.pack(side=tk.LEFT)

        # Информация о графике
        self.plot_info_label = ttk.Label(toolbar_frame, text="График будет показан после загрузки данных",
                                        foreground="gray")
        self.plot_info_label.pack(side=tk.RIGHT, padx=10)

        # Изначально показываем пустой график
        self._show_empty_plot()

    # === МЕТОДЫ РАБОТЫ С ДАННЫМИ ===

    def _auto_load_latest_results(self):
        """Автоматическая загрузка последних результатов кластеризации."""
        try:
            results_dir = Path("clustering_results")
            if not results_dir.exists():
                logger.info("Папка clustering_results не найдена")
                return

            # Поиск последнего CSV файла с результатами
            csv_files = list(results_dir.glob("clustering_results_*.csv"))
            if not csv_files:
                logger.info("CSV файлы с результатами не найдены")
                return

            # Сортировка по времени модификации
            latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)

            logger.info(f"Автозагрузка последнего файла: {latest_csv.name}")
            self.csv_file_var.set(str(latest_csv))
            self._load_csv_file(str(latest_csv))

        except Exception as e:
            logger.error(f"Ошибка автозагрузки: {e}")

    def _browse_csv_file(self):
        """Выбор CSV файла с результатами кластеризации."""
        file_path = filedialog.askopenfilename(
            title="Выберите файл с результатами кластеризации",
            filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")],
            initialdir="clustering_results"
        )

        if file_path:
            self.csv_file_var.set(file_path)
            self._load_csv_file(file_path)

    def _reload_current_file(self):
        """Перезагрузка текущего файла."""
        current_file = self.csv_file_var.get()
        if current_file:
            self._load_csv_file(current_file)
        else:
            messagebox.showwarning("Предупреждение", "Файл не выбран")

    def _load_csv_file(self, file_path: str):
        """
        Загрузка CSV файла с результатами кластеризации.

        :param file_path: Путь к CSV файлу
        """
        try:
            logger.info(f"Загрузка файла: {file_path}")

            # Проверка существования файла
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Файл не найден: {file_path}")

            # Загрузка данных
            self.clustering_data = pd.read_csv(file_path)
            logger.info(f"Загружены данные: {self.clustering_data.shape}")

            # Анализ структуры данных
            self._analyze_loaded_data()

            # Обновление интерфейса
            self._update_k_combobox()
            self._update_plot()
            self._update_statistics()

            # Загрузка метаданных если есть
            self._load_metadata(file_path)

            messagebox.showinfo("Успех", f"Файл загружен успешно!\nДанных: {self.clustering_data.shape[0]} строк, {len(self.available_k_values)} разбиений")

        except Exception as e:
            error_msg = f"Ошибка загрузки файла: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Ошибка", error_msg)

    def _analyze_loaded_data(self):
        """Анализ структуры загруженных данных."""
        logger.info("Анализ структуры данных...")

        # Поиск колонок с признаками (не начинающихся с 'cluster_')
        self.feature_columns = [col for col in self.clustering_data.columns
                               if not col.startswith('cluster_')]

        # Поиск колонок с результатами кластеризации
        cluster_columns = [col for col in self.clustering_data.columns
                          if col.startswith('cluster_k')]

        # Извлечение значений K из названий колонок
        k_values = []
        for col in cluster_columns:
            match = re.search(r'cluster_k(\d+)', col)
            if match:
                k_values.append(int(match.group(1)))

        self.available_k_values = sorted(k_values, reverse=True)

        logger.info(f"Найдены признаки: {self.feature_columns}")
        logger.info(f"Доступные K: {self.available_k_values}")

        # Проверка корректности данных
        if len(self.feature_columns) < 2:
            raise ValueError(f"Недостаточно признаков для визуализации: {len(self.feature_columns)}")

        if len(self.available_k_values) == 0:
            raise ValueError("Не найдены результаты кластеризации")

    def _load_metadata(self, csv_path: str):
        """Загрузка метаданных процесса кластеризации."""
        try:
            # Поиск соответствующего файла метаданных
            csv_file = Path(csv_path)
            results_dir = csv_file.parent

            # Извлекаем timestamp из имени CSV файла
            match = re.search(r'clustering_results_(\d{8}_\d{6})\.csv', csv_file.name)
            if match:
                timestamp = match.group(1)
                metadata_file = results_dir / f"clustering_metadata_{timestamp}.json"

                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
                    logger.info(f"Загружены метаданные: {metadata_file.name}")
                else:
                    logger.info("Файл метаданных не найден")

        except Exception as e:
            logger.warning(f"Не удалось загрузить метаданные: {e}")
            self.metadata = {}

    def _update_k_combobox(self):
        """Обновление списка доступных значений K."""
        if self.available_k_values:
            k_strings = [str(k) for k in self.available_k_values]
            self.k_combobox['values'] = k_strings

            # Устанавливаем значение по умолчанию
            default_k = str(self.available_k_values[0])  # Максимальное K
            if len(self.available_k_values) > len(self.available_k_values) // 2:
                # Если много значений, берем примерно середину
                mid_index = len(self.available_k_values) // 2
                default_k = str(self.available_k_values[mid_index])

            self.selected_k_var.set(default_k)
            self.k_combobox.set(default_k)

    # === МЕТОДЫ ВИЗУАЛИЗАЦИИ ===

    def _show_empty_plot(self):
        """Показ пустого графика с инструкцией."""
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Загрузите файл с результатами\nкластеризации для визуализации',
                    ha='center', va='center', transform=self.ax.transAxes,
                    fontsize=14, alpha=0.5)
        self.ax.set_title('Визуализация кластеризации')
        self.canvas.draw()

    def _show_initial_instructions(self):
        """Показ начальных инструкций в панели статистики."""
        instructions = """
ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ:

1. 📁 ЗАГРУЗКА ДАННЫХ:
   • Нажмите "Авто" для автозагрузки
   • Или выберите CSV файл вручную
   
2. 🎯 ВЫБОР КЛАСТЕРОВ:
   • Выберите K из выпадающего списка
   • График обновится автоматически
   
3. 🎨 НАСТРОЙКА ВИЗУАЛИЗАЦИИ:
   • Измените цветовую схему
   • Настройте размер точек
   • Отрегулируйте прозрачность
   
4. 📊 АНАЛИЗ:
   • Изучите статистику кластеров
   • Используйте навигацию графика
   • Сохраните результаты

ТРЕБОВАНИЯ К ДАННЫМ:
• CSV файл с результатами кластеризации
• Колонки признаков + cluster_kXXX
• Создается ExtendedClusteringProcessor
"""

        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, instructions)
        self.stats_text.config(state=tk.DISABLED)

    def _update_plot(self):
        """Обновление графика кластеризации."""
        if self.clustering_data is None:
            self._show_empty_plot()
            return

        try:
            # Получение параметров визуализации
            selected_k = int(self.selected_k_var.get())
            colormap = self.color_scheme_var.get()
            point_size = self.point_size_var.get()
            alpha = self.transparency_var.get()

            # Получение данных для визуализации
            cluster_column = f'cluster_k{selected_k:03d}'

            if cluster_column not in self.clustering_data.columns:
                raise ValueError(f"Разбиение для K={selected_k} не найдено")

            # Координаты точек (первые два признака)
            x_data = self.clustering_data[self.feature_columns[0]]
            y_data = self.clustering_data[self.feature_columns[1]]
            labels = self.clustering_data[cluster_column]

            # Очистка графика
            self.ax.clear()

            # Построение scatter plot
            scatter = self.ax.scatter(x_data, y_data, c=labels, cmap=colormap,
                                    alpha=alpha, s=point_size, edgecolors='black', linewidth=0.3)

            # Настройка графика
            self.ax.set_xlabel(self.feature_columns[0], fontsize=12)
            self.ax.set_ylabel(self.feature_columns[1], fontsize=12)
            self.ax.set_title(f'Кластеризация: K={selected_k}', fontsize=14, fontweight='bold')
            self.ax.grid(True, alpha=0.3)

            # Добавление цветовой шкалы
            if hasattr(self, 'colorbar'):
                self.colorbar.remove()
            self.colorbar = self.fig.colorbar(scatter, ax=self.ax, shrink=0.8, label='Кластер')

            # Статистика кластеров на графике
            unique_labels = np.unique(labels)
            cluster_counts = [np.sum(labels == label) for label in unique_labels]

            info_text = f"Кластеров: {len(unique_labels)}\nОбъектов: {len(labels)}\nРазмеры: {cluster_counts}"
            self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

            # Обновление информации о графике
            self.plot_info_label.config(text=f"K={selected_k}, точек: {len(labels)}")

            # Обновление canvas
            self.fig.tight_layout()
            self.canvas.draw()

            logger.info(f"График обновлен для K={selected_k}")

        except Exception as e:
            error_msg = f"Ошибка обновления графика: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Ошибка", error_msg)

    def _update_statistics(self):
        """Обновление панели статистики."""
        if self.clustering_data is None:
            return

        try:
            selected_k = int(self.selected_k_var.get())
            cluster_column = f'cluster_k{selected_k:03d}'

            if cluster_column not in self.clustering_data.columns:
                return

            labels = self.clustering_data[cluster_column]
            unique_labels = np.unique(labels)

            # Основная статистика
            stats_content = []
            stats_content.append("=" * 40)
            stats_content.append(f"СТАТИСТИКА КЛАСТЕРИЗАЦИИ (K={selected_k})")
            stats_content.append("=" * 40)
            stats_content.append("")

            # Информация о данных
            stats_content.append("📊 ОБЩАЯ ИНФОРМАЦИЯ:")
            stats_content.append(f"   Всего объектов: {len(labels)}")
            stats_content.append(f"   Количество кластеров: {len(unique_labels)}")
            stats_content.append(f"   Признаки: {', '.join(self.feature_columns[:2])}")
            stats_content.append("")

            # Размеры кластеров
            stats_content.append("📈 РАЗМЕРЫ КЛАСТЕРОВ:")
            cluster_counts = []
            for label in sorted(unique_labels):
                count = np.sum(labels == label)
                percentage = (count / len(labels)) * 100
                cluster_counts.append(count)
                stats_content.append(f"   Кластер {label}: {count} объектов ({percentage:.1f}%)")

            stats_content.append("")
            stats_content.append(f"   Мин. размер: {min(cluster_counts)}")
            stats_content.append(f"   Макс. размер: {max(cluster_counts)}")
            stats_content.append(f"   Средний размер: {np.mean(cluster_counts):.1f}")
            stats_content.append("")

            # Центроиды кластеров
            stats_content.append("🎯 ЦЕНТРОИДЫ КЛАСТЕРОВ:")
            x_col = self.feature_columns[0]
            y_col = self.feature_columns[1]

            for label in sorted(unique_labels):
                cluster_mask = labels == label
                cluster_data = self.clustering_data[cluster_mask]

                centroid_x = cluster_data[x_col].mean()
                centroid_y = cluster_data[y_col].mean()

                stats_content.append(f"   Кластер {label}:")
                stats_content.append(f"     {x_col}: {centroid_x:.3f}")
                stats_content.append(f"     {y_col}: {centroid_y:.3f}")

            # Метаданные процесса
            if self.metadata:
                stats_content.append("")
                stats_content.append("⚙️ ПАРАМЕТРЫ КЛАСТЕРИЗАЦИИ:")
                stats_content.append(f"   Метод: Односвязывающий")
                stats_content.append(f"   Метрика: {self.metadata.get('distance_metric', 'N/A')}")
                stats_content.append(f"   Время выполнения: {self.metadata.get('processing_duration', 'N/A')}")
                if 'available_k_values' in self.metadata:
                    k_range = self.metadata['available_k_values']
                    stats_content.append(f"   Диапазон K: {max(k_range)}→{min(k_range)}")

            # Доступные разбиения
            stats_content.append("")
            stats_content.append("🔢 ДОСТУПНЫЕ РАЗБИЕНИЯ:")
            k_groups = []
            current_group = []
            for i, k in enumerate(self.available_k_values):
                current_group.append(str(k))
                if len(current_group) == 10 or i == len(self.available_k_values) - 1:
                    k_groups.append(", ".join(current_group))
                    current_group = []

            for group in k_groups:
                stats_content.append(f"   {group}")

            # Обновление текстового поля
            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "\n".join(stats_content))
            self.stats_text.config(state=tk.DISABLED)

        except Exception as e:
            logger.error(f"Ошибка обновления статистики: {e}")

    # === ОБРАБОТЧИКИ СОБЫТИЙ ===

    def _on_k_changed(self, event=None):
        """Обработчик изменения выбранного K."""
        try:
            selected_k = int(self.selected_k_var.get())

            # Обновление информации о K
            if self.clustering_data is not None:
                cluster_column = f'cluster_k{selected_k:03d}'
                if cluster_column in self.clustering_data.columns:
                    labels = self.clustering_data[cluster_column]
                    n_clusters = len(np.unique(labels))
                    self.k_info_label.config(text=f"K={selected_k}: {n_clusters} кластеров, {len(labels)} объектов")
                else:
                    self.k_info_label.config(text=f"K={selected_k}: данные не найдены", foreground="red")

            # Обновление графика и статистики
            self._update_plot()
            self._update_statistics()

        except ValueError:
            logger.warning(f"Некорректное значение K: {self.selected_k_var.get()}")

    def _on_visualization_changed(self, event=None):
        """Обработчик изменения настроек визуализации."""
        # Обновление подписей
        self.size_label.config(text=f"Размер: {self.point_size_var.get()}")
        self.alpha_label.config(text=f"Прозрачность: {self.transparency_var.get():.1f}")

        # Обновление графика
        self._update_plot()

    # === МЕТОДЫ ЭКСПОРТА ===

    def _save_plot(self):
        """Сохранение текущего графика."""
        if self.clustering_data is None:
            messagebox.showwarning("Предупреждение", "Нет данных для сохранения")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                title="Сохранить график",
                defaultextension=".png",
                filetypes=[("PNG файлы", "*.png"), ("PDF файлы", "*.pdf"), ("SVG файлы", "*.svg")]
            )

            if file_path:
                selected_k = int(self.selected_k_var.get())
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight',
                               facecolor='white', edgecolor='none')

                messagebox.showinfo("Успех", f"График сохранен: {file_path}")
                logger.info(f"График K={selected_k} сохранен в {file_path}")

        except Exception as e:
            error_msg = f"Ошибка сохранения графика: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Ошибка", error_msg)

    def _export_current_data(self):
        """Экспорт данных для текущего K."""
        if self.clustering_data is None:
            messagebox.showwarning("Предупреждение", "Нет данных для экспорта")
            return

        try:
            selected_k = int(self.selected_k_var.get())
            cluster_column = f'cluster_k{selected_k:03d}'

            # Подготовка данных для экспорта
            export_data = self.clustering_data[self.feature_columns + [cluster_column]].copy()
            export_data = export_data.rename(columns={cluster_column: 'cluster'})

            file_path = filedialog.asksaveasfilename(
                title=f"Экспорт данных K={selected_k}",
                defaultextension=".csv",
                filetypes=[("CSV файлы", "*.csv"), ("Excel файлы", "*.xlsx")]
            )

            if file_path:
                if file_path.endswith('.xlsx'):
                    export_data.to_excel(file_path, index=False)
                else:
                    export_data.to_csv(file_path, index=False)

                messagebox.showinfo("Успех", f"Данные экспортированы: {file_path}")
                logger.info(f"Данные K={selected_k} экспортированы в {file_path}")

        except Exception as e:
            error_msg = f"Ошибка экспорта данных: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Ошибка", error_msg)


def main():
    """Главная функция запуска приложения."""
    root = tk.Tk()
    app = SinglePlotClusteringApp(root)

    # Центрирование окна
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'+{x}+{y}')

    logger.info("Приложение визуализации кластеризации запущено")

    root.mainloop()


if __name__ == "__main__":
    main()
