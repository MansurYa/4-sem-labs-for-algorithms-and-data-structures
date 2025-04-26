import pandas as pd
import numpy as np


def calculate_relative_error(original_df: pd.DataFrame, missing_df: pd.DataFrame, filled_df: pd.DataFrame, rows_removed: bool, column: str) -> float:
    """
    Вычисляет среднюю относительную погрешность для восстановленных значений.

    :param original_df: Исходный DataFrame без пропусков
    :param missing_df: DataFrame с искусственно созданными пропусками
    :param filled_df: DataFrame с заполненными пропусками
    :param rows_removed: Флаг удаления строк
    :param column: Целевой столбец
    :return: Средняя относительная погрешность
    """
    print("original_df:\n", original_df.isnull().sum())

    print("missing_df:\n", missing_df.isnull().sum())

    print("filled_df:\n", filled_df.isnull().sum())

    if rows_removed or column not in original_df.columns:
        return np.nan

    missing_indices = missing_df[missing_df[column].isna()].index

    true_values = original_df.loc[missing_indices, column]

    predicted_values = filled_df.loc[missing_indices, column]

    relative_errors = np.abs((true_values - predicted_values)/true_values) * 100
    return relative_errors.mean()
