import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def check_no_missing_values(df: pd.DataFrame, column: str) -> None:
    """
    Проверяет, что в указанном столбце нет пропущенных значений.

    :param df: pandas DataFrame
    :param column: Название столбца для проверки
    :raises ValueError: Если в столбце есть пропуски
    """
    if df[column].isna().any():
        raise ValueError(f"Столбец '{column}' содержит пропущенные значения.")


def calculate_mean(df: pd.DataFrame, column: str) -> float:
    """
    Вычисляет среднее значение для указанного столбца.

    :param df: pandas DataFrame
    :param column: Название столбца для вычисления
    :return: Среднее значение
    :raises ValueError: Если столбец не числовой или содержит пропуски
    """
    check_no_missing_values(df, column)
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Столбец '{column}' должен быть числовым для вычисления среднего.")
    return df[column].mean()


def calculate_median(df: pd.DataFrame, column: str) -> float:
    """
    Вычисляет медиану для указанного столбца.

    :param df: pandas DataFrame
    :param column: Название столбца для вычисления
    :return: Медианное значение
    :raises ValueError: Если столбец не числовой или содержит пропуски
    """
    check_no_missing_values(df, column)
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Столбец '{column}' должен быть числовым для вычисления медианы.")
    return df[column].median()


def calculate_mode(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Вычисляет моду для указанного столбца.

    :param df: pandas DataFrame
    :param column: Название столбца для вычисления
    :return: pandas Series с модальными значениями
    :raises ValueError: Если столбец содержит пропуски
    """
    check_no_missing_values(df, column)
    return df[column].mode()


def plot_distribution(df: pd.DataFrame, column: str) -> None:
    """
    Строит гистограмму распределения для указанного столбца.

    :param df: pandas DataFrame
    :param column: Название столбца для построения
    :raises ValueError: Если столбец не числовой или содержит пропуски
    """
    check_no_missing_values(df, column)
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Столбец '{column}' должен быть числовым для построения распределения.")
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f"Распределение значений в столбце '{column}'")
    plt.xlabel(column)
    plt.ylabel("Частота")
    plt.show()
