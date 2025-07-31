"""
ИСПРАВЛЕННЫЙ GUI ДЛЯ КЛАСТЕРИЗАЦИИ С РАЗДЕЛЕННЫМ ФУНКЦИОНАЛОМ
Версия 2.1 - КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ:
- Устранен конфликт переменных состояния
- Исправлена логика сравнения кластеризаций
- Добавлены недостающие проверки состояния
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import threading
import time
from pathlib import Path
import logging
from typing import Optional, List, Dict, Any, Tuple

try:
    from mathematical_algorithms import (
        single_linkage_clustering,
        spa_feature_selection,
        fowlkes_mallows_index
    )
    MATH_ALGORITHMS_AVAILABLE = True
except ImportError as e:
    MATH_ALGORITHMS_AVAILABLE = False
    print(f"Предупреждение: Модуль математических алгоритмов недоступен: {e}")

# НОВОЕ: Импорт K-Means функции
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True

    # Добавляем нашу K-Means обертку
    def kmeans_clustering_wrapper(data, n_clusters, metric='euclidean',
                                return_hierarchy=True, verbose=False, random_state=42):
        """Обертка K-Means для совместимости с single_linkage_clustering"""
        import numpy as np

        if metric != 'euclidean' and verbose:
            print(f"K-Means использует только euclidean метрику, игнорируется '{metric}'")

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state,
                       max_iter=300, n_init=10)
        labels = kmeans.fit_predict(data)

        hierarchy = []
        if return_hierarchy:
            centers = kmeans.cluster_centers_
            for i in range(n_clusters):
                cluster_info = {
                    'cluster_id': i,
                    'cluster_center': centers[i].tolist(),
                    'cluster_size': int(np.sum(labels == i)),
                    'inertia': float(kmeans.inertia_)
                }
                hierarchy.append(cluster_info)

        if verbose:
            print(f"K-Means завершен. Inertia: {kmeans.inertia_:.2f}")
            cluster_sizes = [int(np.sum(labels == i)) for i in range(n_clusters)]
            print(f"Размеры кластеров: {cluster_sizes}")

        return (labels, hierarchy) if return_hierarchy else labels

except ImportError as e:
    SKLEARN_AVAILABLE = False
    print(f"Предупреждение: Scikit-learn недоступен: {e}")

    def kmeans_clustering_wrapper(*args, **kwargs):
        raise ImportError("Scikit-learn не установлен. K-Means недоступен.")

try:
    from data_preprocessing import StudentDataProcessor, FeatureSelector
    DATA_PROCESSING_AVAILABLE = True
except ImportError as e:
    DATA_PROCESSING_AVAILABLE = False
    print(f"Предупреждение: Модуль предобработки данных недоступен: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedClusteringApp:
    """
    ИСПРАВЛЕННОЕ приложение для кластеризации с разделенным функционалом.

    КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ v2.1:
    - Переименованы переменные состояния для избежания конфликтов
    - Исправлена логика метода _anonymize_and_compare
    - Добавлены проверки состояния для предотвращения ошибок
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Инструмент анализа кластеризации v2.2 (+ K-Means)")
        self.root.geometry("1400x900")

        # === СОСТОЯНИЕ ДАННЫХ ===
        self.data_processor = None
        self.raw_data = None
        self.processed_data = None

        # === СОСТОЯНИЕ СПА И ВЫБОРА ПРИЗНАКОВ (ПЕРЕИМЕНОВАНЫ ДЛЯ ИЗБЕЖАНИЯ КОНФЛИКТОВ) ===
        self.spa_analysis_results = None  # ИСПРАВЛЕНО: переименовано из spa_results
        self.gui_selected_features = None  # ИСПРАВЛЕНО: переименовано из selected_features
        self.gui_selected_data = None  # ИСПРАВЛЕНО: переименовано из selected_data

        # === СОСТОЯНИЕ ВИЗУАЛИЗАЦИИ ===
        self.visualization_features = None  # 3 признака для визуализации

        # === СОСТОЯНИЕ КЛАСТЕРИЗАЦИИ (СОВМЕСТИМОСТЬ С ОРИГИНАЛЬНЫМ КОДОМ) ===
        self.selected_features = None  # ИСПРАВЛЕНО: Сохранено для совместимости
        self.cluster_labels = None
        self.current_k = 3
        self.clustering_method_used = None  # НОВОЕ: для хранения использованного метода
        self.k_anonymity_results = None
        self.clustering_comparison_results = None

        # === GUI ПЕРЕМЕННЫЕ ===
        self.clustering_method_var = tk.StringVar(value="single_linkage")  # НОВОЕ: выбор метода кластеризации
        self.clustering_metric_var = tk.StringVar(value="euclidean")
        self.quality_metric_var = tk.StringVar(value="euclidean")
        self.k_clusters_var = tk.StringVar(value="3")
        self.n_features_var = tk.StringVar(value="3")
        self.data_file_var = tk.StringVar(value="student_habits_performance.csv")

        # === СОЗДАНИЕ ИНТЕРФЕЙСА ===
        self._create_enhanced_gui_layout()

        # НОВОЕ: Инициализируем обработчик метода кластеризации
        self._on_clustering_method_changed()

        logger.info("Инициализировано Enhanced Clustering App v2.2 (+ K-Means)")

    def _create_enhanced_gui_layout(self):
        """Создание улучшенного макета интерфейса с разделенным функционалом."""

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")

        # === СЕКЦИЯ ДАННЫХ ===
        self._create_data_section(main_frame)

        # === СЕКЦИЯ ПАРАМЕТРОВ ===
        self._create_parameters_section(main_frame)

        # === СЕКЦИЯ УПРАВЛЕНИЯ (РАСШИРЕННАЯ) ===
        self._create_enhanced_controls_section(main_frame)

        # === СЕКЦИЯ РЕЗУЛЬТАТОВ ===
        self._create_results_section(main_frame)

        # === СЕКЦИЯ ПРОГРЕССА ===
        self._create_progress_section(main_frame)

        # === КОНФИГУРАЦИЯ СЕТКИ ===
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)

    def _create_data_section(self, parent):
        """Создание секции управления данными."""
        data_frame = ttk.LabelFrame(parent, text="📊 Управление данными", padding=5)
        data_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)

        ttk.Label(data_frame, text="Файл данных:").grid(row=0, column=0, sticky="w")
        file_entry = ttk.Entry(data_frame, textvariable=self.data_file_var,
                              state="readonly", width=60)
        file_entry.grid(row=0, column=1, padx=5)
        ttk.Button(data_frame, text="Обзор...",
                  command=self._browse_file).grid(row=0, column=2)

    def _create_parameters_section(self, parent):
        """Создание секции параметров."""
        params_frame = ttk.LabelFrame(parent, text="⚙️ Параметры кластеризации", padding=5)
        params_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)

        # Строка 1: Метод кластеризации (НОВОЕ)
        ttk.Label(params_frame, text="Метод кластеризации:").grid(row=0, column=0, sticky="w")

        # НОВОЕ: Определяем доступные методы на основе установленных модулей
        available_methods = []
        if MATH_ALGORITHMS_AVAILABLE:
            available_methods.append("single_linkage")
        if SKLEARN_AVAILABLE:
            available_methods.append("kmeans")

        # Если нет доступных методов, показываем ошибку
        if not available_methods:
            available_methods = ["Нет доступных методов"]

        method_combo = ttk.Combobox(params_frame, textvariable=self.clustering_method_var,
                                   values=available_methods,
                                   state="readonly", width=15)
        method_combo.grid(row=0, column=1, padx=5)
        method_combo.bind('<<ComboboxSelected>>', self._on_clustering_method_changed)  # Обработчик изменения

        # Устанавливаем значение по умолчанию
        if available_methods and "Нет доступных методов" not in available_methods:
            self.clustering_method_var.set(available_methods[0])

        ttk.Label(params_frame, text="Количество кластеров (K):").grid(row=0, column=2, sticky="w", padx=10)
        ttk.Spinbox(params_frame, from_=2, to=20, textvariable=self.k_clusters_var,
                   width=8).grid(row=0, column=3, padx=5)

        # Строка 2: Метрика расстояния
        ttk.Label(params_frame, text="Метрика расстояния:").grid(row=1, column=0, sticky="w", pady=5)
        self.metric_combo = ttk.Combobox(params_frame, textvariable=self.clustering_metric_var,
                                        values=["euclidean", "euclidean_squared", "chebyshev"],
                                        state="readonly", width=15)
        self.metric_combo.grid(row=1, column=1, padx=5, pady=5)

        # Строка 3: Параметры СПА
        ttk.Label(params_frame, text="Признаков для СПА:").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Spinbox(params_frame, from_=3, to=15, textvariable=self.n_features_var,
                   width=8).grid(row=2, column=1, padx=5, pady=5)

    def _create_enhanced_controls_section(self, parent):
        """Создание расширенной секции управления с разделенным функционалом."""
        controls_frame = ttk.LabelFrame(parent, text="🎛️ Управление процессом", padding=5)
        controls_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)

        # === ЭТАП 1: ПОДГОТОВКА ДАННЫХ ===
        stage1_frame = ttk.Frame(controls_frame)
        stage1_frame.pack(fill=tk.X, pady=2)

        ttk.Label(stage1_frame, text="ЭТАП 1: Подготовка данных",
                 font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W)

        stage1_buttons = ttk.Frame(stage1_frame)
        stage1_buttons.pack(fill=tk.X, pady=2)

        self.load_btn = ttk.Button(stage1_buttons, text="📂 Загрузить данные",
                                  command=self._load_data)
        self.preprocess_btn = ttk.Button(stage1_buttons, text="⚙️ Предобработать",
                                       state="disabled", command=self._preprocess_data)

        self.load_btn.pack(side=tk.LEFT, padx=2)
        self.preprocess_btn.pack(side=tk.LEFT, padx=2)

        # === ЭТАП 2: АНАЛИЗ ПРИЗНАКОВ (РАЗДЕЛЕННЫЙ ФУНКЦИОНАЛ) ===
        stage2_frame = ttk.Frame(controls_frame)
        stage2_frame.pack(fill=tk.X, pady=2)

        ttk.Label(stage2_frame, text="ЭТАП 2: Анализ и выбор признаков",
                 font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W)

        stage2_buttons = ttk.Frame(stage2_frame)
        stage2_buttons.pack(fill=tk.X, pady=2)

        # РАЗДЕЛЕННЫЕ КНОПКИ
        self.spa_btn = ttk.Button(stage2_buttons, text="🔍 Запустить СПА",
                                 state="disabled", command=self._run_spa_analysis)
        self.apply_features_btn = ttk.Button(stage2_buttons, text="✂️ Применить выбор признаков",
                                           state="disabled", command=self._apply_feature_selection)

        self.spa_btn.pack(side=tk.LEFT, padx=2)
        self.apply_features_btn.pack(side=tk.LEFT, padx=2)

        # === ЭТАП 3: КЛАСТЕРИЗАЦИЯ ===
        stage3_frame = ttk.Frame(controls_frame)
        stage3_frame.pack(fill=tk.X, pady=2)

        ttk.Label(stage3_frame, text="ЭТАП 3: Кластеризация",
                 font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W)

        stage3_buttons = ttk.Frame(stage3_frame)
        stage3_buttons.pack(fill=tk.X, pady=2)

        self.cluster_btn = ttk.Button(stage3_buttons, text="🎯 Кластеризовать",
                                    state="disabled", command=self._perform_clustering)
        self.viz_select_btn = ttk.Button(stage3_buttons, text="👁️ Выбрать признаки для визуализации",
                                       state="disabled", command=self._select_visualization_features)

        self.cluster_btn.pack(side=tk.LEFT, padx=2)
        self.viz_select_btn.pack(side=tk.LEFT, padx=2)

        # === ЭТАП 4: АНАЛИЗ ===
        stage4_frame = ttk.Frame(controls_frame)
        stage4_frame.pack(fill=tk.X, pady=2)

        ttk.Label(stage4_frame, text="ЭТАП 4: Анализ результатов",
                 font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W)

        stage4_buttons = ttk.Frame(stage4_frame)
        stage4_buttons.pack(fill=tk.X, pady=2)

        self.analyze_btn = ttk.Button(stage4_buttons, text="📊 Анализ K-анонимности",
                                    state="disabled", command=self._analyze_k_anonymity)
        self.anonymize_btn = ttk.Button(stage4_buttons, text="🔒 Анонимизировать и сравнить",
                                      state="disabled", command=self._anonymize_and_compare)

        self.analyze_btn.pack(side=tk.LEFT, padx=2)
        self.anonymize_btn.pack(side=tk.LEFT, padx=2)

    def _create_results_section(self, parent):
        """Создание секции результатов с улучшенной визуализацией."""
        results_frame = ttk.LabelFrame(parent, text="📈 Результаты анализа", padding=5)
        results_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=5)

        # === ПАНЕЛЬ ВИЗУАЛИЗАЦИИ ===
        viz_frame = ttk.Frame(results_frame)
        viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # === ПАНЕЛЬ СТАТИСТИКИ (РАСШИРЕННАЯ) ===
        stats_outer_frame = ttk.Frame(results_frame)
        stats_outer_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)

        # Notebook для разных типов статистики
        self.stats_notebook = ttk.Notebook(stats_outer_frame)
        self.stats_notebook.pack(fill=tk.BOTH, expand=True)

        # Вкладка: Общая статистика
        general_frame = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(general_frame, text="Общая статистика")

        self.stats_text = tk.Text(general_frame, wrap=tk.WORD, width=45, height=15)
        stats_scrollbar = ttk.Scrollbar(general_frame, command=self.stats_text.yview)
        self.stats_text.config(yscrollcommand=stats_scrollbar.set)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Вкладка: Результаты СПА
        spa_frame = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(spa_frame, text="Результаты СПА")

        self.spa_text = tk.Text(spa_frame, wrap=tk.WORD, width=45, height=15)
        spa_scrollbar = ttk.Scrollbar(spa_frame, command=self.spa_text.yview)
        self.spa_text.config(yscrollcommand=spa_scrollbar.set)
        self.spa_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        spa_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Вкладка: Состояние данных
        data_frame = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(data_frame, text="Состояние данных")

        self.data_text = tk.Text(data_frame, wrap=tk.WORD, width=45, height=15)
        data_scrollbar = ttk.Scrollbar(data_frame, command=self.data_text.yview)
        self.data_text.config(yscrollcommand=data_scrollbar.set)
        self.data_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        data_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_progress_section(self, parent):
        """Создание секции прогресса."""
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(parent, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=5, column=0, columnspan=2, sticky="ew", pady=5)

        self.status_label = ttk.Label(parent, text="Готово к работе v2.2 (+ K-Means)")
        self.status_label.grid(row=6, column=0, columnspan=2, sticky="w")

    # ===============================
    # МЕТОДЫ ЗАГРУЗКИ И ПРЕДОБРАБОТКИ
    # ===============================

    def _browse_file(self):
        """Выбор файла данных."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV файлы", "*.csv")])
        if file_path:
            self.data_file_var.set(file_path)

    def _load_data(self):
        """Загрузка данных с обновленной логикой."""
        self._run_with_progress(self._load_data_worker, "Загрузка данных...")

    def _load_data_worker(self, progress_callback):
        """Воркер для загрузки данных."""
        try:
            progress_callback(20, "Инициализация процессора данных...")
            self.data_processor = StudentDataProcessor(self.data_file_var.get())

            progress_callback(60, "Загрузка и анализ данных...")
            self.raw_data = self.data_processor.load_and_analyze_data()

            progress_callback(100, "Данные загружены!")
            self.root.after(0, self._update_buttons_after_load)
            self.root.after(0, self._update_data_status)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error loading data: {error_msg}")
            self.root.after(0, lambda: messagebox.showerror("Ошибка", error_msg))

    def _preprocess_data(self):
        """Предобработка данных."""
        self._run_with_progress(self._preprocess_data_worker, "Предобработка данных...")

    def _preprocess_data_worker(self, progress_callback):
        """Воркер для предобработки данных."""
        try:
            progress_callback(30, "Выполнение предобработки...")
            self.processed_data = self.data_processor.preprocess_data()

            progress_callback(100, "Предобработка завершена!")
            self.root.after(0, self._update_buttons_after_preprocess)
            self.root.after(0, self._update_data_status)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error preprocessing data: {error_msg}")
            self.root.after(0, lambda: messagebox.showerror("Ошибка", error_msg))

    # ========================================
    # НОВЫЙ МЕТОД: ОБРАБОТЧИК ИЗМЕНЕНИЯ МЕТОДА КЛАСТЕРИЗАЦИИ
    # ========================================

    def _on_clustering_method_changed(self, event=None):
        """Обработчик изменения метода кластеризации."""
        method = self.clustering_method_var.get()

        # НОВОЕ: Проверяем доступность методов
        if method == "kmeans" and not SKLEARN_AVAILABLE:
            messagebox.showerror("Ошибка", "Scikit-learn недоступен. K-Means метод не может быть использован.")
            # Переключаемся на single_linkage если доступен
            if MATH_ALGORITHMS_AVAILABLE:
                self.clustering_method_var.set("single_linkage")
                method = "single_linkage"
            else:
                return

        elif method == "single_linkage" and not MATH_ALGORITHMS_AVAILABLE:
            messagebox.showerror("Ошибка", "Математические алгоритмы недоступны. Односвязывающий метод не может быть использован.")
            # Переключаемся на kmeans если доступен
            if SKLEARN_AVAILABLE:
                self.clustering_method_var.set("kmeans")
                method = "kmeans"
            else:
                return

        if method == "kmeans":
            # Для K-Means доступна только euclidean метрика
            self.metric_combo['values'] = ["euclidean"]
            self.clustering_metric_var.set("euclidean")
            self.metric_combo.config(state="disabled")
        else:  # single_linkage
            # Для односвязывающего метода доступны все метрики
            self.metric_combo['values'] = ["euclidean", "euclidean_squared", "chebyshev"]
            self.metric_combo.config(state="readonly")

        # Обновляем статус в интерфейсе
        self._update_method_info_display()

    def _update_method_info_display(self):
        """Обновление информации о выбранном методе кластеризации."""
        method = self.clustering_method_var.get()

        method_info = {
            "kmeans": "K-Means: Быстрый алгоритм разбиения. Только euclidean метрика.",
            "single_linkage": "Односвязывающий: Иерархический метод. Поддерживает разные метрики."
        }

        info = method_info.get(method, "")
        self.status_label.config(text=info, foreground="blue")

    # ========================================
    # ИСПРАВЛЕННЫЕ МЕТОДЫ ДЛЯ РАЗДЕЛЕННОГО ФУНКЦИОНАЛА
    # ========================================

    def _run_spa_analysis(self):
        """Запуск СПА анализа без применения результатов к датасету."""
        self._run_with_progress(self._spa_analysis_worker, "Запуск СПА анализа...")

    def _spa_analysis_worker(self, progress_callback):
        """Воркер для выполнения СПА анализа."""
        try:
            n_features = int(self.n_features_var.get())
            progress_callback(20, "Инициализация СПА...")

            # Запускаем СПА, но НЕ применяем результаты к датасету
            feature_selector = FeatureSelector(random_state=42)
            selected_features, quality = feature_selector.select_features_spa(
                self.processed_data,
                n_features=n_features,
                clustering_method='single_linkage',
                n_clusters=int(self.k_clusters_var.get()),
                distance_metric=self.clustering_metric_var.get()
            )

            # ИСПРАВЛЕНО: Сохраняем результаты СПА в переименованную переменную
            self.spa_analysis_results = {
                'selected_features': selected_features,
                'quality_score': quality,
                'feature_selector': feature_selector,
                'all_features': list(self.processed_data.columns),
                'n_features_requested': n_features
            }

            progress_callback(100, "СПА анализ завершен!")
            self.root.after(0, self._update_buttons_after_spa)
            self.root.after(0, self._update_spa_results_display)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in SPA analysis: {error_msg}")
            self.root.after(0, lambda: messagebox.showerror("Ошибка", error_msg))

    def _apply_feature_selection(self):
        """Применение выбора признаков к датасету."""
        if self.spa_analysis_results is None:
            messagebox.showwarning("Предупреждение",
                                 "Сначала запустите СПА анализ для получения рекомендаций")
            return

        # Создаем диалог для выбора признаков
        self._show_feature_selection_dialog()

    def _show_feature_selection_dialog(self):
        """Показ диалога для выбора признаков."""
        dialog = FeatureSelectionDialog(self.root, self.spa_analysis_results, self.processed_data.columns)
        result = dialog.result

        if result:
            selected_features = result['selected_features']

            # ИСПРАВЛЕНО: Применяем выбор признаков к GUI переменным
            self.gui_selected_features = selected_features
            self.gui_selected_data = self.processed_data[selected_features].copy()

            # ИСПРАВЛЕНО: Также сохраняем в переменную совместимости для метода сравнения
            self.selected_features = selected_features

            # Обновляем интерфейс
            self._update_buttons_after_feature_selection()
            self._update_data_status()

            logger.info(f"Применен выбор признаков: {selected_features}")

    def _select_visualization_features(self):
        """Выбор 3 признаков для визуализации."""
        # ИСПРАВЛЕНО: Определяем доступные признаки с проверкой состояния
        if self.gui_selected_data is not None:
            available_features = list(self.gui_selected_data.columns)
            data_for_viz = self.gui_selected_data
        else:
            available_features = list(self.processed_data.columns)
            data_for_viz = self.processed_data

        if len(available_features) < 3:
            messagebox.showerror("Ошибка",
                               "Недостаточно признаков для 3D визуализации. Нужно минимум 3 признака.")
            return

        # Создаем диалог для выбора признаков визуализации
        dialog = VisualizationFeaturesDialog(self.root, available_features)
        result = dialog.result

        if result:
            self.visualization_features = result['selected_features']
            self._update_visualization()
            logger.info(f"Выбраны признаки для визуализации: {self.visualization_features}")

    # ===============================
    # ИСПРАВЛЕННЫЕ МЕТОДЫ КЛАСТЕРИЗАЦИИ
    # ===============================

    def _perform_clustering(self):
        """ИСПРАВЛЕННЫЙ метод кластеризации с улучшенными проверками."""
        if self.processed_data is None:
            messagebox.showerror("Ошибка", "Сначала выполните предобработку данных")
            return

        try:
            k_clusters = int(self.k_clusters_var.get())
            if k_clusters < 2:
                messagebox.showerror("Ошибка", "Количество кластеров должно быть минимум 2")
                return
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректное количество кластеров")
            return

        # ИСПРАВЛЕНО: Улучшенная логика выбора данных с проверками
        if self.gui_selected_data is not None and not self.gui_selected_data.empty:
            data_for_clustering = self.gui_selected_data
            # ИСПРАВЛЕНО: Безопасная проверка наличия gui_selected_features
            if self.gui_selected_features is not None:
                data_source = f"выбранные признаки ({len(self.gui_selected_features)})"
            else:
                data_source = f"выбранные данные ({self.gui_selected_data.shape[1]} признаков)"
        else:
            data_for_clustering = self.processed_data
            data_source = f"все предобработанные признаки ({self.processed_data.shape[1]})"

        self._run_with_progress(
            lambda progress_callback: self._clustering_worker(
                progress_callback, data_for_clustering, data_source
            ),
            "Выполнение кластеризации..."
        )

    def _clustering_worker(self, progress_callback, data_for_clustering, data_source):
        """ИСПРАВЛЕННЫЙ воркер кластеризации с поддержкой двух методов."""
        try:
            k_clusters = int(self.k_clusters_var.get())
            clustering_metric = self.clustering_metric_var.get()
            clustering_method = self.clustering_method_var.get()  # НОВОЕ: получаем выбранный метод

            progress_callback(30, f"Подготовка данных ({data_source})...")

            # ИСПРАВЛЕНО: Дополнительная проверка данных
            if data_for_clustering.empty:
                raise ValueError("Данные для кластеризации пусты")

            if data_for_clustering.shape[0] < k_clusters:
                raise ValueError(f"Количество объектов ({data_for_clustering.shape[0]}) меньше количества кластеров ({k_clusters})")

            # НОВОЕ: Выбор метода кластеризации
            if clustering_method == "kmeans":
                if not SKLEARN_AVAILABLE:
                    raise ImportError("Scikit-learn не установлен. K-Means недоступен.")

                progress_callback(70, "Выполнение K-Means кластеризации...")
                self.cluster_labels, self.cluster_hierarchy = kmeans_clustering_wrapper(
                    data_for_clustering.values,
                    n_clusters=k_clusters,
                    metric=clustering_metric,
                    return_hierarchy=True,
                    verbose=False,
                    random_state=42
                )

            elif clustering_method == "single_linkage":
                if not MATH_ALGORITHMS_AVAILABLE:
                    raise ImportError("Математические алгоритмы недоступны")

                progress_callback(70, "Выполнение односвязывающего метода...")
                self.cluster_labels, self.cluster_hierarchy = single_linkage_clustering(
                    data_for_clustering.values,
                    n_clusters=k_clusters,
                    metric=clustering_metric,
                    return_hierarchy=True,
                    verbose=False
                )
            else:
                raise ValueError(f"Неизвестный метод кластеризации: {clustering_method}")

            # Сохраняем информацию о методе
            self.current_k = k_clusters
            self.data_source = data_source
            self.clustering_method_used = clustering_method  # НОВОЕ: сохраняем использованный метод

            progress_callback(100, "Кластеризация завершена!")
            self.root.after(0, self._update_visualization)
            self.root.after(0, self._update_statistics)
            self.root.after(0, self._update_buttons_after_clustering)

            logger.info(f"Кластеризация завершена методом {clustering_method} на {data_source}: {k_clusters} кластеров")

        except Exception as e:
            error_msg = f"Ошибка при кластеризации: {str(e)}"
            logger.error(error_msg)
            self.root.after(0, lambda: messagebox.showerror("Ошибка", error_msg))

    # ===============================
    # ИСПРАВЛЕННЫЕ МЕТОДЫ ОБНОВЛЕНИЯ ВИЗУАЛИЗАЦИИ
    # ===============================

    def _update_visualization(self):
        """ИСПРАВЛЕННЫЙ метод визуализации с улучшенной логикой выбора признаков."""
        self.fig.clear()

        if self.cluster_labels is None:
            # Показываем пустой график
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Выполните кластеризацию\nдля отображения результатов',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, alpha=0.6)
            ax.set_title('Результаты кластеризации')
            self.canvas.draw()
            return

        # ИСПРАВЛЕНО: Улучшенная логика определения данных для визуализации
        if self.visualization_features and len(self.visualization_features) >= 2:
            # Используем вручную выбранные признаки для визуализации
            if self.gui_selected_data is not None:
                # Проверяем, есть ли выбранные признаки в урезанном датасете
                available_viz_features = [f for f in self.visualization_features
                                        if f in self.gui_selected_data.columns]
                if len(available_viz_features) >= 2:
                    viz_data = self.gui_selected_data[available_viz_features[:3]]
                    viz_features = available_viz_features[:3]
                else:
                    # Fallback: используем первые признаки из урезанного датасета
                    viz_data = self.gui_selected_data.iloc[:, :min(3, self.gui_selected_data.shape[1])]
                    viz_features = list(viz_data.columns)
            else:
                # Используем признаки из полного датасета
                viz_data = self.processed_data[self.visualization_features[:3]]
                viz_features = self.visualization_features[:3]
        else:
            # Fallback: используем первые доступные признаки
            if self.gui_selected_data is not None:
                viz_data = self.gui_selected_data.iloc[:, :min(3, self.gui_selected_data.shape[1])]
                viz_features = list(viz_data.columns)
            else:
                viz_data = self.processed_data.iloc[:, :min(3, self.processed_data.shape[1])]
                viz_features = list(viz_data.columns)

        # ИСПРАВЛЕНО: Дополнительная проверка данных для визуализации
        if viz_data.empty:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Нет данных для визуализации',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, alpha=0.6)
            ax.set_title('Ошибка визуализации')
            self.canvas.draw()
            return

        # Создаем визуализацию
        if len(viz_features) >= 3:
            # 3D визуализация
            ax = self.fig.add_subplot(111, projection='3d')
            x = viz_data.iloc[:, 0]
            y = viz_data.iloc[:, 1]
            z = viz_data.iloc[:, 2]

            scatter = ax.scatter(x, y, z, c=self.cluster_labels, cmap='viridis', alpha=0.6)
            ax.set_xlabel(viz_features[0])
            ax.set_ylabel(viz_features[1])
            ax.set_zlabel(viz_features[2])
            ax.set_title(f'Кластеризация (K={self.current_k})')

            # Добавляем цветовую шкалу
            self.fig.colorbar(scatter, ax=ax, shrink=0.8, label='Кластер')

        elif len(viz_features) >= 2:
            # 2D визуализация
            ax = self.fig.add_subplot(111)
            x = viz_data.iloc[:, 0]
            y = viz_data.iloc[:, 1]

            scatter = ax.scatter(x, y, c=self.cluster_labels, cmap='viridis', alpha=0.6)
            ax.set_xlabel(viz_features[0])
            ax.set_ylabel(viz_features[1])
            ax.set_title(f'Кластеризация (K={self.current_k})')
            ax.grid(True, alpha=0.3)

            # Добавляем цветовую шкалу
            self.fig.colorbar(scatter, ax=ax, shrink=0.8, label='Кластер')

        self.canvas.draw()

    # ===============================
    # ИСПРАВЛЕННЫЕ МЕТОДЫ ОБНОВЛЕНИЯ СТАТИСТИКИ
    # ===============================

    def _update_statistics(self):
        """Обновление статистики с поддержкой нового функционала."""
        self._update_general_statistics()
        self._update_data_status()

    def _update_general_statistics(self):
        """ОБНОВЛЕННОЕ обновление общей статистики с информацией о методе."""
        stats = []

        if self.cluster_labels is not None:
            stats.append("=== РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ ===")

            # НОВОЕ: Отображение метода кластеризации
            method_used = getattr(self, 'clustering_method_used', 'неизвестно')
            method_names = {
                'kmeans': 'K-Means',
                'single_linkage': 'Односвязывающий метод'
            }
            method_display = method_names.get(method_used, method_used)

            stats.append(f"Метод кластеризации: {method_display}")
            stats.append(f"Количество кластеров: {self.current_k}")
            stats.append(f"Метрика расстояния: {self.clustering_metric_var.get()}")
            stats.append(f"Источник данных: {getattr(self, 'data_source', 'неизвестно')}")

            # Статистика кластеров
            unique_labels = np.unique(self.cluster_labels)
            stats.append(f"\nРаспределение по кластерам:")
            for label in unique_labels:
                count = np.sum(self.cluster_labels == label)
                percentage = (count / len(self.cluster_labels)) * 100
                stats.append(f"  Кластер {label}: {count} объектов ({percentage:.1f}%)")

            # НОВОЕ: Дополнительная статистика для K-Means
            if method_used == 'kmeans' and hasattr(self, 'cluster_hierarchy') and self.cluster_hierarchy:
                try:
                    # Извлекаем информацию из иерархии K-Means
                    total_inertia = self.cluster_hierarchy[0].get('inertia', 0)
                    stats.append(f"\nДополнительная статистика K-Means:")
                    stats.append(f"  Общая внутрикластерная сумма квадратов: {total_inertia:.2f}")

                    # Центроиды кластеров
                    stats.append(f"  Координаты центроидов:")
                    for cluster_info in self.cluster_hierarchy:
                        cluster_id = cluster_info.get('cluster_id', 'N/A')
                        center = cluster_info.get('cluster_center', [])
                        if len(center) <= 3:  # Показываем только первые 3 координаты
                            center_str = ', '.join([f"{c:.3f}" for c in center])
                            stats.append(f"    Кластер {cluster_id}: [{center_str}]")
                        else:
                            center_str = ', '.join([f"{c:.3f}" for c in center[:3]]) + "..."
                            stats.append(f"    Кластер {cluster_id}: [{center_str}]")
                except Exception as e:
                    stats.append(f"  Ошибка извлечения статистики K-Means: {e}")

        if self.k_anonymity_results:
            stats.append("\n=== K-АНОНИМНОСТЬ ===")
            stats.append(f"Значение K: {self.k_anonymity_results.get('k_anonymity_value', 'Н/Д')}")

        if self.clustering_comparison_results:
            stats.append("\n=== СРАВНЕНИЕ МЕТОДОВ ===")
            fm_indices = self.clustering_comparison_results.get('fm_indices', {})
            stats.append("Индекс Фоулкса-Мэллова:")
            for comparison, value in fm_indices.items():
                stats.append(f"  {comparison}: {value:.4f}")

        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, "\n".join(stats))
        self.stats_text.config(state=tk.DISABLED)

    def _update_spa_results_display(self):
        """ИСПРАВЛЕННОЕ обновление отображения результатов СПА."""
        if self.spa_analysis_results is None:
            content = "СПА анализ не выполнен"
        else:
            content = []
            content.append("=== РЕЗУЛЬТАТЫ СПА АНАЛИЗА ===")
            content.append(f"Запрошено признаков: {self.spa_analysis_results['n_features_requested']}")
            content.append(f"Качество выбора: {self.spa_analysis_results['quality_score']:.6f}")
            content.append(f"\nРекомендуемые признаки:")

            for i, feature in enumerate(self.spa_analysis_results['selected_features'], 1):
                content.append(f"  {i}. {feature}")

            content.append(f"\nВсего доступно признаков: {len(self.spa_analysis_results['all_features'])}")
            content.append(f"\nВсе доступные признаки:")
            for i, feature in enumerate(self.spa_analysis_results['all_features'], 1):
                marker = "⭐" if feature in self.spa_analysis_results['selected_features'] else "  "
                content.append(f"{marker} {i:2d}. {feature}")

        self.spa_text.config(state=tk.NORMAL)
        self.spa_text.delete(1.0, tk.END)
        self.spa_text.insert(tk.END, "\n".join(content))
        self.spa_text.config(state=tk.DISABLED)

    def _update_data_status(self):
        """ИСПРАВЛЕННОЕ обновление статуса данных."""
        status = []

        # Состояние исходных данных
        if self.raw_data is not None:
            status.append("=== ИСХОДНЫЕ ДАННЫЕ ===")
            if hasattr(self.data_processor, 'raw_data') and self.data_processor.raw_data is not None:
                status.append(f"Размер: {self.data_processor.raw_data.shape}")
            else:
                status.append("Загружены")

        # Состояние предобработанных данных
        if self.processed_data is not None:
            status.append("\n=== ПРЕДОБРАБОТАННЫЕ ДАННЫЕ ===")
            status.append(f"Размер: {self.processed_data.shape}")
            status.append(f"Тип предобработки: StandardScaler (μ=0, σ=1)")  # ДОБАВЛЕНО
            status.append(f"Признаки: {list(self.processed_data.columns)}")

        # Состояние СПА
        if self.spa_analysis_results is not None:
            status.append("\n=== РЕЗУЛЬТАТЫ СПА ===")
            status.append(f"Статус: Выполнен")
            status.append(f"Рекомендуемые признаки: {self.spa_analysis_results['selected_features']}")

        # ИСПРАВЛЕНО: Состояние выбранных данных
        if self.gui_selected_data is not None:
            status.append("\n=== ВЫБРАННЫЕ ДАННЫЕ ===")
            status.append(f"Размер: {self.gui_selected_data.shape}")
            status.append(f"Признаки: {self.gui_selected_features}")

        # Состояние визуализации
        if self.visualization_features is not None:
            status.append("\n=== ПРИЗНАКИ ДЛЯ ВИЗУАЛИЗАЦИИ ===")
            status.append(f"Выбраны: {self.visualization_features}")

        self.data_text.config(state=tk.NORMAL)
        self.data_text.delete(1.0, tk.END)
        self.data_text.insert(tk.END, "\n".join(status))
        self.data_text.config(state=tk.DISABLED)

    # ===============================
    # КРИТИЧЕСКИ ИСПРАВЛЕННЫЙ МЕТОД СРАВНЕНИЯ
    # ===============================

    def _anonymize_and_compare(self):
        """КРИТИЧЕСКИ ИСПРАВЛЕННЫЙ метод анонимизации и сравнения."""
        # ИСПРАВЛЕНО: Проверяем состояние данных перед сравнением
        if self.processed_data is None:
            messagebox.showerror("Ошибка", "Сначала выполните предобработку данных")
            return

        if self.cluster_labels is None:
            messagebox.showerror("Ошибка", "Сначала выполните кластеризацию")
            return

        self._run_with_progress(self._anonymize_worker_fixed, "Анонимизация и сравнение...")

    def _anonymize_worker_fixed(self, progress_callback):
        """КРИТИЧЕСКИ ИСПРАВЛЕННЫЙ воркер для анонимизации."""
        try:
            k_clusters = int(self.k_clusters_var.get())
            metric = self.clustering_metric_var.get()

            progress_callback(20, "Подготовка к сравнению...")

            # ИСПРАВЛЕНО: Определяем количество признаков на основе текущего состояния
            if self.gui_selected_features is not None:
                # Если пользователь выбрал признаки, используем их количество
                n_features = len(self.gui_selected_features)
                logger.info(f"Используем количество признаков из GUI выбора: {n_features}")
            else:
                # Если признаки не выбраны, используем параметр СПА
                n_features = int(self.n_features_var.get())
                logger.info(f"Используем параметр СПА для количества признаков: {n_features}")

            progress_callback(40, "Запуск сравнения кластеризаций...")

            # ИСПРАВЛЕНО: Передаем корректное количество признаков
            results = self.data_processor.compare_clustering_results(
                k_clusters=k_clusters,
                n_features=n_features,
                metric=metric,
                anonymization_level='moderate'
            )

            self.clustering_comparison_results = results

            progress_callback(100, "Анализ завершен!")
            self.root.after(0, self._update_statistics)

            # ИСПРАВЛЕНО: Улучшенное отображение результатов
            if 'fm_indices' in results:
                fm_indices = results['fm_indices']
                message = f"""Результаты сравнения кластеризаций:

Индекс Фоулкса-Мэллова:
• Предобработка vs Выбор признаков: {fm_indices.get('preprocessing_vs_features', 'Н/Д'):.4f}
• Предобработка vs Анонимизация: {fm_indices.get('preprocessing_vs_anonymized', 'Н/Д'):.4f}  
• Выбор признаков vs Анонимизация: {fm_indices.get('features_vs_anonymized', 'Н/Д'):.4f}

Использовано признаков для сравнения: {n_features}"""

                self.root.after(0, lambda: messagebox.showinfo("Результаты сравнения", message))
            else:
                self.root.after(0, lambda: messagebox.showerror("Ошибка", "Не удалось получить результаты сравнения"))

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in anonymization: {error_msg}")
            self.root.after(0, lambda: messagebox.showerror("Ошибка", error_msg))

    # ===============================
    # МЕТОДЫ УПРАВЛЕНИЯ СОСТОЯНИЕМ КНОПОК (ИСПРАВЛЕНЫ)
    # ===============================

    def _update_buttons_after_load(self):
        """Обновление состояния кнопок после загрузки данных."""
        self.preprocess_btn.config(state="normal")

    def _update_buttons_after_preprocess(self):
        """Обновление состояния кнопок после предобработки."""
        self.spa_btn.config(state="normal")
        self.cluster_btn.config(state="normal")  # Можно кластеризовать без СПА
        self.analyze_btn.config(state="normal")

    def _update_buttons_after_spa(self):
        """Обновление состояния кнопок после СПА анализа."""
        self.apply_features_btn.config(state="normal")

    def _update_buttons_after_feature_selection(self):
        """Обновление состояния кнопок после применения выбора признаков."""
        # Все кнопки остаются активными
        pass

    def _update_buttons_after_clustering(self):
        """Обновление состояния кнопок после кластеризации."""
        self.viz_select_btn.config(state="normal")
        self.anonymize_btn.config(state="normal")

    # ===============================
    # ОСТАЛЬНЫЕ МЕТОДЫ (БЕЗ ИЗМЕНЕНИЙ)
    # ===============================

    def _analyze_k_anonymity(self):
        """Анализ K-анонимности (без изменений)."""
        self._run_with_progress(self._k_anonymity_worker, "Анализ K-анонимности...")

    def _k_anonymity_worker(self, progress_callback):
        """Воркер для анализа K-анонимности (без изменений)."""
        try:
            progress_callback(50, "Анализ...")
            self.k_anonymity_results = self.data_processor.analyze_k_anonymity()

            progress_callback(100, "Анализ завершён!")
            self.root.after(0, self._update_statistics)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error analyzing k-anonymity: {error_msg}")
            self.root.after(0, lambda: messagebox.showerror("Ошибка", error_msg))

    # ===============================
    # СЛУЖЕБНЫЕ МЕТОДЫ (БЕЗ ИЗМЕНЕНИЙ)
    # ===============================

    def _run_with_progress(self, worker, message):
        """Запуск операции с индикатором прогресса (без изменений)."""
        self._disable_all_buttons()
        self.status_label.config(text=message, foreground="blue")
        self.progress_var.set(0)

        def thread_wrapper():
            try:
                worker(lambda v, s=None: self._update_progress(v, s))
                self.root.after(0, self._operation_complete)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Operation failed: {error_msg}")
                self.root.after(0, lambda: self._operation_failed(error_msg))

        threading.Thread(target=thread_wrapper, daemon=True).start()

    def _update_progress(self, value, status=None):
        """Обновление прогресса (без изменений)."""
        self.progress_var.set(value)
        if status:
            self.status_label.config(text=status)
        self.root.update_idletasks()

    def _operation_complete(self):
        """Завершение операции (без изменений)."""
        self.status_label.config(text="Операция завершена!", foreground="green")
        self._enable_appropriate_buttons()

    def _operation_failed(self, error):
        """Обработка ошибки операции (без изменений)."""
        self.status_label.config(text=f"Ошибка: {error}", foreground="red")
        self._enable_appropriate_buttons()
        messagebox.showerror("Ошибка", error)

    def _disable_all_buttons(self):
        """Отключение всех кнопок (обновлено)."""
        for btn in [self.load_btn, self.preprocess_btn, self.spa_btn,
                   self.apply_features_btn, self.cluster_btn, self.viz_select_btn,
                   self.analyze_btn, self.anonymize_btn]:
            btn.config(state="disabled")

    def _enable_appropriate_buttons(self):
        """ИСПРАВЛЕННОЕ включение соответствующих кнопок в зависимости от состояния."""
        # Всегда доступные
        self.load_btn.config(state="normal")

        # После загрузки данных
        if self.raw_data is not None:
            self.preprocess_btn.config(state="normal")
            self.analyze_btn.config(state="normal")

        # После предобработки
        if self.processed_data is not None:
            self.spa_btn.config(state="normal")
            self.cluster_btn.config(state="normal")  # Можно кластеризовать без СПА

        # После СПА
        if self.spa_analysis_results is not None:
            self.apply_features_btn.config(state="normal")

        # После кластеризации
        if self.cluster_labels is not None:
            self.viz_select_btn.config(state="normal")
            self.anonymize_btn.config(state="normal")  # ИСПРАВЛЕНО: упрощено условие


# ===============================
# ДИАЛОГОВЫЕ ОКНА (БЕЗ ИЗМЕНЕНИЙ)
# ===============================

class FeatureSelectionDialog:
    """Диалоговое окно для выбора признаков с рекомендациями СПА."""

    def __init__(self, parent, spa_results, all_features):
        self.result = None

        # Создаем модальное окно
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Выбор признаков для применения")
        self.dialog.geometry("800x600")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Центрируем окно
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (800 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (600 // 2)
        self.dialog.geometry(f"+{x}+{y}")

        self._create_dialog_content(spa_results, all_features)

        # Ожидаем закрытие диалога
        self.dialog.wait_window()

    def _create_dialog_content(self, spa_results, all_features):
        """Создание содержимого диалога."""

        # Заголовок
        title_frame = ttk.Frame(self.dialog)
        title_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(title_frame, text="Выбор признаков для применения к датасету",
                 font=('TkDefaultFont', 12, 'bold')).pack()

        # Информация о СПА
        info_frame = ttk.LabelFrame(self.dialog, text="Рекомендации СПА", padding=5)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        spa_info = tk.Text(info_frame, height=6, wrap=tk.WORD)
        spa_info.pack(fill=tk.X)

        spa_text = f"""Алгоритм СПА рекомендует следующие {len(spa_results['selected_features'])} признаков:
{', '.join(spa_results['selected_features'])}

Качество выбора: {spa_results['quality_score']:.6f}

Вы можете принять эти рекомендации или выбрать признаки самостоятельно."""

        spa_info.insert(tk.END, spa_text)
        spa_info.config(state=tk.DISABLED)

        # Выбор признаков
        selection_frame = ttk.LabelFrame(self.dialog, text="Выбор признаков", padding=5)
        selection_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Создаем Checkbuttons для каждого признака
        self.feature_vars = {}

        # Создаем Canvas с прокруткой
        canvas = tk.Canvas(selection_frame)
        scrollbar = ttk.Scrollbar(selection_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Добавляем чекбоксы
        for i, feature in enumerate(all_features):
            var = tk.BooleanVar()
            # По умолчанию выбираем рекомендованные СПА признаки
            if feature in spa_results['selected_features']:
                var.set(True)

            self.feature_vars[feature] = var

            # Добавляем маркер для рекомендованных признаков
            text = f"⭐ {feature}" if feature in spa_results['selected_features'] else f"   {feature}"

            cb = ttk.Checkbutton(scrollable_frame, text=text, variable=var)
            cb.grid(row=i, column=0, sticky="w", padx=5, pady=1)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Кнопки управления
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="Выбрать все",
                  command=self._select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Снять все",
                  command=self._deselect_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="СПА рекомендации",
                  command=lambda: self._select_spa_features(spa_results['selected_features'])).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Применить",
                  command=self._apply_selection).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Отмена",
                  command=self._cancel).pack(side=tk.RIGHT, padx=5)

    def _select_all(self):
        """Выбрать все признаки."""
        for var in self.feature_vars.values():
            var.set(True)

    def _deselect_all(self):
        """Снять выбор со всех признаков."""
        for var in self.feature_vars.values():
            var.set(False)

    def _select_spa_features(self, spa_features):
        """Выбрать только рекомендованные СПА признаки."""
        for feature, var in self.feature_vars.items():
            var.set(feature in spa_features)

    def _apply_selection(self):
        """Применить выбор."""
        selected = [feature for feature, var in self.feature_vars.items() if var.get()]

        if not selected:
            messagebox.showwarning("Предупреждение", "Выберите хотя бы один признак")
            return

        self.result = {'selected_features': selected}
        self.dialog.destroy()

    def _cancel(self):
        """Отмена."""
        self.result = None
        self.dialog.destroy()


class VisualizationFeaturesDialog:
    """Диалоговое окно для выбора 3 признаков для визуализации."""

    def __init__(self, parent, available_features):
        self.result = None

        # Создаем модальное окно
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Выбор признаков для визуализации")
        self.dialog.geometry("500x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Центрируем окно
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (500 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (400 // 2)
        self.dialog.geometry(f"+{x}+{y}")

        self._create_viz_dialog_content(available_features)

        # Ожидаем закрытие диалога
        self.dialog.wait_window()

    def _create_viz_dialog_content(self, available_features):
        """Создание содержимого диалога визуализации."""

        # Заголовок
        title_frame = ttk.Frame(self.dialog)
        title_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(title_frame, text="Выбор признаков для 3D визуализации",
                 font=('TkDefaultFont', 12, 'bold')).pack()
        ttk.Label(title_frame, text="Выберите 3 признака для отображения на осях X, Y, Z",
                 font=('TkDefaultFont', 9)).pack()

        # Выбор признаков
        selection_frame = ttk.LabelFrame(self.dialog, text="Признаки", padding=10)
        selection_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # X ось
        ttk.Label(selection_frame, text="Ось X:").grid(row=0, column=0, sticky="w", pady=5)
        self.x_var = tk.StringVar(value=available_features[0] if available_features else "")
        x_combo = ttk.Combobox(selection_frame, textvariable=self.x_var,
                              values=available_features, state="readonly", width=30)
        x_combo.grid(row=0, column=1, padx=10, pady=5)

        # Y ось
        ttk.Label(selection_frame, text="Ось Y:").grid(row=1, column=0, sticky="w", pady=5)
        self.y_var = tk.StringVar(value=available_features[1] if len(available_features) > 1 else "")
        y_combo = ttk.Combobox(selection_frame, textvariable=self.y_var,
                              values=available_features, state="readonly", width=30)
        y_combo.grid(row=1, column=1, padx=10, pady=5)

        # Z ось
        ttk.Label(selection_frame, text="Ось Z:").grid(row=2, column=0, sticky="w", pady=5)
        self.z_var = tk.StringVar(value=available_features[2] if len(available_features) > 2 else "")
        z_combo = ttk.Combobox(selection_frame, textvariable=self.z_var,
                              values=available_features, state="readonly", width=30)
        z_combo.grid(row=2, column=1, padx=10, pady=5)

        # Информация
        info_frame = ttk.Frame(self.dialog)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        info_text = """💡 Совет: Выбирайте признаки с разными диапазонами значений
для лучшего разделения кластеров на графике."""

        ttk.Label(info_frame, text=info_text,
                 font=('TkDefaultFont', 8), foreground="gray").pack()

        # Кнопки
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="Применить",
                  command=self._apply_viz_selection).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Отмена",
                  command=self._cancel_viz).pack(side=tk.RIGHT, padx=5)

    def _apply_viz_selection(self):
        """Применить выбор признаков для визуализации."""
        x_feature = self.x_var.get()
        y_feature = self.y_var.get()
        z_feature = self.z_var.get()

        if not all([x_feature, y_feature, z_feature]):
            messagebox.showwarning("Предупреждение", "Выберите все три признака")
            return

        if len(set([x_feature, y_feature, z_feature])) != 3:
            messagebox.showwarning("Предупреждение", "Признаки должны быть разными")
            return

        self.result = {'selected_features': [x_feature, y_feature, z_feature]}
        self.dialog.destroy()

    def _cancel_viz(self):
        """Отмена выбора визуализации."""
        self.result = None
        self.dialog.destroy()


def main():
    """Главная функция запуска исправленного приложения."""
    root = tk.Tk()
    app = EnhancedClusteringApp(root)

    # Центрирование окна
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'+{x}+{y}')

    if not MATH_ALGORITHMS_AVAILABLE and not SKLEARN_AVAILABLE:
        messagebox.showerror("Критическая ошибка",
            "Отсутствуют необходимые модули для кластеризации!\n"
            "Требуется либо mathematical_algorithms, либо scikit-learn.")
        return
    elif not MATH_ALGORITHMS_AVAILABLE:
        messagebox.showwarning("Предупреждение",
            "Модуль mathematical_algorithms недоступен. Односвязывающий метод отключен.\n"
            "Доступен только K-Means.")
    elif not SKLEARN_AVAILABLE:
        messagebox.showwarning("Предупреждение",
            "Scikit-learn недоступен. K-Means отключен.\n"
            "Доступен только односвязывающий метод.")

    if not DATA_PROCESSING_AVAILABLE:
        messagebox.showwarning("Предупреждение",
            "Модуль предобработки данных отсутствует. Функциональность ограничена.")

    logger.info("Enhanced Clustering App v2.2 (+ K-Means) запущено успешно")
    root.mainloop()


if __name__ == "__main__":
    main()
