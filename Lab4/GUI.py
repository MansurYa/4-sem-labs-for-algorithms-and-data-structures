import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nan_processing import remove_contiguous_values, remove_rows_with_missing_values, fill_missing_with_mean, fill_missing_with_regression
from statistics_functions import calculate_mean, calculate_median, calculate_mode, plot_distribution
from relative_error import calculate_relative_error


class DataProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Обработка пропусков в данных")

        self.original_df = None
        self.missing_df = None
        self.filled_df = None
        self.missing_indices = None

        self.input_frame = ttk.Frame(root)
        self.original_stats_frame = ttk.Frame(root)
        self.filled_stats_frame = ttk.Frame(root)

        self.input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")
        self.original_stats_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")
        self.filled_stats_frame.grid(row=0, column=2, padx=10, pady=10, sticky="n")

        self.create_input_widgets()
        self.create_stats_widgets()

    def create_input_widgets(self):
        ttk.Label(self.input_frame, text="Выберите датасет:").grid(row=0, column=0, sticky="w")
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(self.input_frame, textvariable=self.dataset_var, state="readonly")
        self.dataset_combo["values"] = ["purchases_data_2000.csv", "purchases_data_20000.csv", "purchases_data_200000.csv", "purchases_data_2000000.csv"]
        self.dataset_combo.grid(row=0, column=1)
        self.dataset_combo.bind("<<ComboboxSelected>>", self.load_dataset)

        ttk.Label(self.input_frame, text="Выберите столбец:").grid(row=1, column=0, sticky="w")
        self.column_var = tk.StringVar()
        self.column_combo = ttk.Combobox(self.input_frame, textvariable=self.column_var, state="readonly")
        self.column_combo["values"] = ["Долгота", "Широта", "Количество товаров", "Стоимость"]
        self.column_combo.set("Стоимость")
        self.column_combo.grid(row=1, column=1)

        ttk.Label(self.input_frame, text="Процент удаления (%):").grid(row=2, column=0, sticky="w")
        self.percent_var = tk.StringVar()
        self.percent_combo = ttk.Combobox(self.input_frame, textvariable=self.percent_var, state="readonly")
        self.percent_combo["values"] = ["3", "5", "10", "20", "30"]
        self.percent_combo.grid(row=2, column=1)

        ttk.Label(self.input_frame, text="Метод восстановления:").grid(row=3, column=0, sticky="w")
        self.method_var = tk.StringVar()
        self.method_combo = ttk.Combobox(self.input_frame, textvariable=self.method_var, state="readonly")
        self.method_combo["values"] = ["Удаление строк", "Заполнение средним", "Заполнение регрессией"]
        self.method_combo.grid(row=3, column=1)

        self.plot_button = ttk.Button(self.input_frame, text="Построить гистограмму", command=self.plot_histogram)
        self.plot_button.grid(row=4, column=0, columnspan=2, pady=5)

        self.remove_button = ttk.Button(self.input_frame, text="Удалить значения", command=self.remove_values)
        self.remove_button.grid(row=5, column=0, columnspan=2, pady=5)

        self.fill_button = ttk.Button(self.input_frame, text="Восстановить значения", command=self.fill_values)
        self.fill_button.grid(row=6, column=0, columnspan=2, pady=5)

    def create_stats_widgets(self):
        ttk.Label(self.original_stats_frame, text="Статистики исходного df").grid(row=0, column=0, columnspan=2)
        self.original_mean_label = ttk.Label(self.original_stats_frame, text="Среднее: -")
        self.original_mean_label.grid(row=1, column=0, sticky="w")
        self.original_median_label = ttk.Label(self.original_stats_frame, text="Медиана: -")
        self.original_median_label.grid(row=2, column=0, sticky="w")
        self.original_mode_label = ttk.Label(self.original_stats_frame, text="Мода: -")
        self.original_mode_label.grid(row=3, column=0, sticky="w")

        ttk.Label(self.filled_stats_frame, text="Статистики восстановленного df").grid(row=0, column=0, columnspan=2)
        self.filled_mean_label = ttk.Label(self.filled_stats_frame, text="Среднее: -")
        self.filled_mean_label.grid(row=1, column=0, sticky="w")
        self.filled_median_label = ttk.Label(self.filled_stats_frame, text="Медиана: -")
        self.filled_median_label.grid(row=2, column=0, sticky="w")
        self.filled_mode_label = ttk.Label(self.filled_stats_frame, text="Мода: -")
        self.filled_mode_label.grid(row=3, column=0, sticky="w")
        self.filled_error_label = ttk.Label(self.filled_stats_frame, text="Относительная погрешность: -")
        self.filled_error_label.grid(row=4, column=0, sticky="w")

    def load_dataset(self, event):
        dataset = self.dataset_var.get()
        self.original_df = pd.read_csv(dataset)
        self.missing_df = None
        self.filled_df = None
        self.missing_indices = None
        self.update_original_stats()

    def update_original_stats(self):
        column = self.column_var.get()
        if self.original_df is not None and column in self.original_df.columns:
            mean = calculate_mean(self.original_df, column)
            median = calculate_median(self.original_df, column)
            mode = calculate_mode(self.original_df, column)
            self.original_mean_label.config(text=f"Среднее: {mean:.2f}")
            self.original_median_label.config(text=f"Медиана: {median:.2f}")
            self.original_mode_label.config(text=f"Мода: {', '.join(map(str, mode))}")

    def remove_values(self):
        if self.original_df is None:
            return
        column = self.column_var.get()
        percent = float(self.percent_var.get()) / 100
        self.missing_df = remove_contiguous_values(self.original_df.copy(), column, percent)
        self.missing_indices = self.missing_df.index[self.missing_df[column].isna()]
        self.filled_df = None
        self.reset_filled_stats()

    def fill_values(self):
        if self.missing_df is None:
            return
        column = self.column_var.get()
        method = self.method_var.get()
        rows_removed = False
        if method == "Удаление строк":
            original_rows = self.missing_df.shape[0]
            self.filled_df = remove_rows_with_missing_values(self.missing_df.copy(), column)
            removed_rows = original_rows - self.filled_df.shape[0]
            rows_removed = removed_rows > 0
            if self.filled_df.empty:
                print("Ошибка: после удаления строк данных не осталось.")
            else:
                print(f"Удалено {removed_rows} строк с пропусками в столбце '{column}'")
        elif method == "Заполнение средним":
            self.filled_df = fill_missing_with_mean(self.missing_df.copy(), column)
        elif method == "Заполнение регрессией":
            self.filled_df = fill_missing_with_regression(self.missing_df.copy(), column)
        self.update_filled_stats(rows_removed)

    def reset_filled_stats(self):
        self.filled_mean_label.config(text="Среднее: -")
        self.filled_median_label.config(text="Медиана: -")
        self.filled_mode_label.config(text="Мода: -")
        self.filled_error_label.config(text="Относительная погрешность: -")

    def update_filled_stats(self, rows_removed=False):
        column = self.column_var.get()
        if self.filled_df is not None and column in self.filled_df.columns:
            mean = calculate_mean(self.filled_df, column)
            median = calculate_median(self.filled_df, column)
            mode = calculate_mode(self.filled_df, column)
            self.filled_mean_label.config(text=f"Среднее: {mean:.2f}")
            self.filled_median_label.config(text=f"Медиана: {median:.2f}")
            self.filled_mode_label.config(text=f"Мода: {', '.join(map(str, mode))}")

            try:
                relative_error = calculate_relative_error(original_df=self.original_df, missing_df=self.missing_df, filled_df=self.filled_df, rows_removed=rows_removed, column=column)
                if pd.isna(relative_error):
                    error_text = "N/A"
                elif relative_error == 0.0:
                    error_text = "0.00%"
                else:
                    error_text = f"{relative_error:.2f}%"
            except ValueError as e:
                error_text = f"Ошибка: {str(e)}"
            self.filled_error_label.config(text=f"Относительная погрешность: {error_text}")

    def plot_histogram(self):
        column = self.column_var.get()
        if self.filled_df is not None:
            df = self.filled_df
            title = "Распределение восстановленного df"
        elif self.missing_df is not None:
            df = self.missing_df
            title = "Распределение df с удалениями"
        else:
            df = self.original_df
            title = "Распределение исходного df"

        if df is not None and column in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[column], kde=True)
            plt.title(f"{title} для столбца '{column}'")
            plt.xlabel(column)
            plt.ylabel("Частота")
            plt.show()
