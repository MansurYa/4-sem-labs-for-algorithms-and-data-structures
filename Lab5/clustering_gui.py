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

try:
    from data_preprocessing import StudentDataProcessor, FeatureSelector
    DATA_PROCESSING_AVAILABLE = True
except ImportError as e:
    DATA_PROCESSING_AVAILABLE = False
    print(f"Предупреждение: Модуль предобработки данных недоступен: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClusteringApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Инструмент анализа кластеризации")
        self.root.geometry("1200x800")

        # State variables
        self.data_processor = None
        self.raw_data = None
        self.processed_data = None
        self.selected_features = None
        self.cluster_labels = None
        self.current_k = 3
        self.k_anonymity_results = None

        # GUI variables
        self.clustering_metric_var = tk.StringVar(value="euclidean")
        self.quality_metric_var = tk.StringVar(value="euclidean")
        self.k_clusters_var = tk.StringVar(value="3")
        self.n_features_var = tk.StringVar(value="3")
        self.data_file_var = tk.StringVar(value="student_habits_performance.csv")

        self._create_gui_layout()

    def _create_gui_layout(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Data section
        data_frame = ttk.LabelFrame(main_frame, text="Данные", padding=5)
        data_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)

        ttk.Label(data_frame, text="Файл данных:").grid(row=0, column=0, sticky="w")
        file_entry = ttk.Entry(data_frame, textvariable=self.data_file_var, state="readonly", width=50)
        file_entry.grid(row=0, column=1, padx=5)
        ttk.Button(data_frame, text="Обзор...", command=self._browse_file).grid(row=0, column=2)

        # Parameters section
        params_frame = ttk.LabelFrame(main_frame, text="Параметры", padding=5)
        params_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)

        ttk.Label(params_frame, text="Метрика кластеризации:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(params_frame, textvariable=self.clustering_metric_var,
                    values=["euclidean", "euclidean_squared", "chebyshev"], state="readonly").grid(row=0, column=1)

        ttk.Label(params_frame, text="Количество кластеров (K):").grid(row=0, column=2, sticky="w", padx=10)
        ttk.Spinbox(params_frame, from_=2, to=20, textvariable=self.k_clusters_var, width=5).grid(row=0, column=3)

        ttk.Label(params_frame, text="Количество признаков:").grid(row=0, column=4, sticky="w", padx=10)
        ttk.Spinbox(params_frame, from_=3, to=15, textvariable=self.n_features_var, width=5).grid(row=0, column=5)

        # Controls section
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=2, column=0, columnspan=2, pady=5)

        self.load_btn = ttk.Button(controls_frame, text="Загрузить данные", command=self._load_data)
        self.preprocess_btn = ttk.Button(controls_frame, text="Предобработать", state="disabled", command=self._preprocess_data)
        self.features_btn = ttk.Button(controls_frame, text="Выбрать признаки", state="disabled", command=self._select_features)
        self.cluster_btn = ttk.Button(controls_frame, text="Кластеризовать", state="disabled", command=self._perform_clustering)
        self.analyze_btn = ttk.Button(controls_frame, text="Анализ K-анонимности", state="disabled", command=self._analyze_k_anonymity)
        # кнопка
        self.anonymize_btn = ttk.Button(controls_frame, text="Анонимизировать и сравнить", state="disabled", command=self._anonymize_and_compare)

        self.load_btn.pack(side=tk.LEFT, padx=5)
        self.preprocess_btn.pack(side=tk.LEFT, padx=5)
        self.features_btn.pack(side=tk.LEFT, padx=5)
        self.cluster_btn.pack(side=tk.LEFT, padx=5)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        self.anonymize_btn.pack(side=tk.LEFT, padx=5)

        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Результаты", padding=5)
        results_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=5)

        # Visualization
        self.fig = plt.Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=results_frame)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Statistics
        stats_frame = ttk.Frame(results_frame)
        stats_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)

        self.stats_text = tk.Text(stats_frame, wrap=tk.WORD, width=40, height=20)
        scrollbar = ttk.Scrollbar(stats_frame, command=self.stats_text.yview)
        self.stats_text.config(yscrollcommand=scrollbar.set)

        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)
        self.status_label = ttk.Label(main_frame, text="Готово")
        self.status_label.grid(row=5, column=0, columnspan=2, sticky="w")

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)

    def _browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV файлы", "*.csv")])
        if file_path:
            self.data_file_var.set(file_path)

    def _load_data(self):
        self._run_with_progress(self._load_data_worker, "Загрузка данных...")

    def _load_data_worker(self, progress_callback):
        try:
            progress_callback(20, "Инициализация процессора данных...")
            self.data_processor = StudentDataProcessor(self.data_file_var.get())

            progress_callback(60, "Загрузка данных...")
            self.raw_data = self.data_processor.load_and_analyze_data()

            progress_callback(100, "Данные загружены!")
            self.root.after(0, self._enable_preprocess_button)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error loading data: {error_msg}")
            self.root.after(0, lambda: messagebox.showerror("Ошибка", error_msg))

    def _preprocess_data(self):
        self._run_with_progress(self._preprocess_data_worker, "Предобработка данных...")

    def _preprocess_data_worker(self, progress_callback):
        try:
            progress_callback(30, "Предобработка...")
            self.processed_data = self.data_processor.preprocess_data()

            progress_callback(100, "Предобработка завершена!")
            self.root.after(0, self._enable_features_button)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error preprocessing data: {error_msg}")
            self.root.after(0, lambda: messagebox.showerror("Ошибка", error_msg))

    def _select_features(self):
        self._run_with_progress(self._select_features_worker, "Выбор признаков...")

    def _select_features_worker(self, progress_callback):
        try:
            n_features = int(self.n_features_var.get())
            progress_callback(20, "Запуск алгоритма SPA...")
            self.selected_features = self.data_processor.select_informative_features(n_features)

            progress_callback(100, "Признаки выбраны!")
            self.root.after(0, self._enable_cluster_button)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error selecting features: {error_msg}")
            self.root.after(0, lambda: messagebox.showerror("Ошибка", error_msg))

    def _perform_clustering(self):
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

        if self.selected_features is not None:
            data_for_clustering = self.data_processor.get_selected_data()
            data_source = "выбранные признаки"
        else:
            data_for_clustering = self.processed_data
            data_source = "все предобработанные данные"

        self._run_with_progress(
            lambda progress_callback: self._clustering_worker(progress_callback, data_for_clustering, data_source),
            "Выполнение кластеризации..."
        )

    def _clustering_worker(self, progress_callback, data_for_clustering, data_source):
        try:
            k_clusters = int(self.k_clusters_var.get())
            clustering_metric = self.clustering_metric_var.get()

            # Этап 1: Подготовка данных
            progress_callback(30, f"Подготовка данных для кластеризации ({data_source})...")

            # Проверка доступности алгоритмов
            if not MATH_ALGORITHMS_AVAILABLE:
                raise ImportError("Математические алгоритмы недоступны")

            # Этап 2: Кластеризация
            progress_callback(70, "Выполнение односвязывающего метода...")
            self.cluster_labels, self.cluster_hierarchy = single_linkage_clustering(
                data_for_clustering.values,
                n_clusters=k_clusters,
                metric=clustering_metric,
                return_hierarchy=True,
                verbose=False
            )

            # Сохранение результатов
            self.current_k = k_clusters
            self.data_source = data_source

            # Этап 3: Визуализация
            progress_callback(100, "Создание визуализации...")
            self.root.after(0, self._update_visualization)
            self.root.after(0, self._update_statistics)
            self.root.after(0, self._enable_buttons)

            logger.info(f"Кластеризация завершена на {data_source}: {k_clusters} кластеров")

        except Exception as e:
            error_msg = f"Ошибка при кластеризации: {str(e)}"
            logger.error(error_msg)
            self.root.after(0, lambda: messagebox.showerror("Ошибка", error_msg))

    def _generate_statistics_content(self) -> str:
        content_parts = []

        # Информация о кластеризации
        if self.cluster_labels is not None:
            content_parts.append(f"Кластеризация (на {self.data_source}):")
            content_parts.append(f"  Метод: Односвязывающий")
            content_parts.append(f"  Метрика: {self.clustering_metric_var.get()}")
            content_parts.append(f"  Количество кластеров: {self.current_k}")
            # (Дополнительная статистика кластеров может быть добавлена здесь)

        return "\n".join(content_parts)

    def _analyze_k_anonymity(self):
        self._run_with_progress(self._k_anonymity_worker, "Анализ K-анонимности...")

    def _k_anonymity_worker(self, progress_callback):
        try:
            progress_callback(50, "Анализ...")
            self.k_anonymity_results = self.data_processor.analyze_k_anonymity()

            progress_callback(100, "Анализ завершён!")
            self.root.after(0, self._update_statistics)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error analyzing k-anonymity: {error_msg}")
            self.root.after(0, lambda: messagebox.showerror("Ошибка", error_msg))

    def _update_visualization(self):
        self.fig.clear()
        if self.selected_features and len(self.selected_features) >= 3:
            ax = self.fig.add_subplot(111, projection='3d')
            data = self.data_processor.get_selected_data()
            x = data.iloc[:, 0]
            y = data.iloc[:, 1]
            z = data.iloc[:, 2]

            ax.scatter(x, y, z, c=self.cluster_labels, cmap='viridis')
            ax.set_xlabel(self.selected_features[0])
            ax.set_ylabel(self.selected_features[1])
            ax.set_zlabel(self.selected_features[2])
        self.canvas.draw()

    def _update_statistics(self):
        stats = []
        if self.cluster_labels is not None:
            stats.append("=== Результаты кластеризации ===")
            stats.append(f"Кластеры: {self.current_k}")
            stats.append(f"Метрика: {self.clustering_metric_var.get()}")

        if self.k_anonymity_results:
            stats.append("\n=== K-анонимность ===")
            stats.append(f"Значение K: {self.k_anonymity_results.get('k_anonymity_value', 'Н/Д')}")

        if hasattr(self, 'clustering_comparison_results') and self.clustering_comparison_results:
            stats.append("\n=== Сравнение методов кластеризации ===")
            fm_indices = self.clustering_comparison_results.get('fm_indices', {})
            stats.append(f"Индекс Фоулкса-Мэллова:")
            stats.append(f"  Предобработка vs Выбор признаков: {fm_indices.get('preprocessing_vs_features', 'Н/Д'):.4f}")
            stats.append(f"  Предобработка vs Анонимизация: {fm_indices.get('preprocessing_vs_anonymized', 'Н/Д'):.4f}")
            stats.append(f"  Выбор признаков vs Анонимизация: {fm_indices.get('features_vs_anonymized', 'Н/Д'):.4f}")

        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, "\n".join(stats))
        self.stats_text.config(state=tk.DISABLED)

    def _run_with_progress(self, worker, message):
        self._disable_buttons()
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
        self.progress_var.set(value)
        if status:
            self.status_label.config(text=status)
        self.root.update_idletasks()

    def _operation_complete(self):
        self.status_label.config(text="Операция завершена!", foreground="green")
        self._enable_buttons()

    def _operation_failed(self, error):
        self.status_label.config(text=f"Ошибка: {error}", foreground="red")
        self._enable_buttons()
        messagebox.showerror("Ошибка", error)

    def _disable_buttons(self):
        for btn in [self.load_btn, self.preprocess_btn, self.features_btn,
                   self.cluster_btn, self.analyze_btn, self.anonymize_btn]:
            btn.config(state="disabled")

    def _enable_buttons(self):
        states = {
            'load': True,
            'preprocess': self.raw_data is not None,
            'features': self.processed_data is not None,
            'cluster': self.processed_data is not None,
            'analyze': self.raw_data is not None,
            'anonymize': self.processed_data is not None and self.selected_features is not None
        }
        # Применяем состояния к кнопкам
        self.load_btn.config(state="normal")
        self.preprocess_btn.config(state="normal" if states['preprocess'] else "disabled")
        self.features_btn.config(state="normal" if states['features'] else "disabled")
        self.cluster_btn.config(state="normal" if states['cluster'] else "disabled")
        self.analyze_btn.config(state="normal" if states['analyze'] else "disabled")
        self.anonymize_btn.config(state="normal" if states['anonymize'] else "disabled")

    def _enable_preprocess_button(self):
        self.preprocess_btn.config(state="normal")

    def _enable_features_button(self):
        self.features_btn.config(state="normal")

    def _enable_cluster_button(self):
        self.cluster_btn.config(state="normal")

    def compare_clustering_results(self, k_clusters=3, n_features=5, metric='euclidean',
                                  anonymization_level='moderate'):
        """
        Сравнивает результаты различных подходов к кластеризации,
        вычисляя индекс Фоулкса-Мэллова между ними.
        Улучшенная версия с более надежной обработкой ошибок.
        """
        try:
            from mathematical_algorithms import fowlkes_mallows_index, single_linkage_clustering
        except ImportError as e:
            logger.error(f"Не удалось импортировать необходимые функции: {e}")
            return {"error": "Не удалось импортировать необходимые функции"}

        results = {}

        try:
            # 1. Кластеризация после предобработки
            logger.info("Выполняем кластеризацию предобработанных данных")
            if self.processed_data is None:
                self.preprocess_data()

            labels_preprocessing = single_linkage_clustering(
                self.processed_data.values,
                n_clusters=k_clusters,
                metric=metric,
                return_hierarchy=False,
                verbose=False
            )
            results["labels_preprocessing"] = labels_preprocessing

            # 2. Кластеризация после выбора признаков
            logger.info("Выполняем кластеризацию с выбором признаков")
            if self.selected_features is None:
                self.select_informative_features(n_features=n_features)

            selected_data = self.get_selected_data()
            labels_features = single_linkage_clustering(
                selected_data.values,
                n_clusters=k_clusters,
                metric=metric,
                return_hierarchy=False,
                verbose=False
            )
            results["labels_features"] = labels_features

            # 3. Кластеризация после анонимизации
            logger.info("Выполняем кластеризацию анонимизированных данных")
            try:
                labels_anonymized, _ = self.anonymize_and_cluster(
                    n_features=n_features,
                    k_clusters=k_clusters,
                    metric=metric,
                    anonymization_level=anonymization_level
                )
                results["labels_anonymized"] = labels_anonymized
            except Exception as e:
                logger.error(f"Ошибка при анонимизации и кластеризации: {e}")
                # Устанавливаем фиктивные метки для возможности продолжения
                results["labels_anonymized"] = np.zeros_like(labels_preprocessing)
                results["anonymization_error"] = str(e)

            # Вычисляем индексы Фоулкса-Мэллова для всех пар результатов
            fm_preproc_vs_features = fowlkes_mallows_index(labels_preprocessing, labels_features)

            if "anonymization_error" not in results:
                fm_preproc_vs_anon = fowlkes_mallows_index(labels_preprocessing, labels_anonymized)
                fm_features_vs_anon = fowlkes_mallows_index(labels_features, labels_anonymized)
            else:
                fm_preproc_vs_anon = 0.0
                fm_features_vs_anon = 0.0

            results["fm_indices"] = {
                "preprocessing_vs_features": fm_preproc_vs_features,
                "preprocessing_vs_anonymized": fm_preproc_vs_anon,
                "features_vs_anonymized": fm_features_vs_anon
            }

            logger.info(f"Результаты сравнения кластеризации: {results['fm_indices']}")

        except Exception as e:
            logger.error(f"Ошибка при сравнении методов кластеризации: {e}")
            results["error"] = str(e)

        return results

    def _anonymize_and_compare(self):
        self._run_with_progress(self._anonymize_worker, "Анонимизация и сравнение...")

    def _anonymize_worker(self, progress_callback):
        try:
            k_clusters = int(self.k_clusters_var.get())
            n_features = int(self.n_features_var.get())
            metric = self.clustering_metric_var.get()

            progress_callback(20, "Запуск анонимизации и сравнения...")

            results = self.data_processor.compare_clustering_results(
                k_clusters=k_clusters,
                n_features=n_features,
                metric=metric,
                anonymization_level='moderate'
            )

            self.clustering_comparison_results = results

            progress_callback(100, "Анализ завершен!")
            self.root.after(0, self._update_statistics)
            self.root.after(0, lambda: messagebox.showinfo("Результаты сравнения",
                                   f"Индекс Фоулкса-Мэллова:\n"
                                   f"Предобработка vs Выбор признаков: {results['fm_indices']['preprocessing_vs_features']:.4f}\n"
                                   f"Предобработка vs Анонимизация: {results['fm_indices']['preprocessing_vs_anonymized']:.4f}\n"
                                   f"Выбор признаков vs Анонимизация: {results['fm_indices']['features_vs_anonymized']:.4f}"))

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in anonymization: {error_msg}")
            self.root.after(0, lambda: messagebox.showerror("Ошибка", error_msg))

def main():
    root = tk.Tk()
    app = ClusteringApp(root)

    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'+{x}+{y}')

    if not MATH_ALGORITHMS_AVAILABLE or not DATA_PROCESSING_AVAILABLE:
        messagebox.showwarning("Предупреждение",
            "Некоторые необходимые модули отсутствуют. Функциональность ограничена.")

    root.mainloop()


if __name__ == "__main__":
    main()
