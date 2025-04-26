import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def remove_values(df: pd.DataFrame, column: str, percent: float) -> pd.DataFrame:
    """
    Создает новый DataFrame, в котором удалены случайные значения в указанном столбце в заданном проценте.
    Исходный DataFrame остается неизменным.

    :param df: Исходный pandas DataFrame
    :param column: Название столбца (str)
    :param percent: Доля значений для удаления (float, 0 < percent < 1)
    :return: Новый DataFrame с удаленными значениями
    :raises ValueError: Если percent не в диапазоне (0, 1) или column имеет неверный тип
    """
    if not 0 < percent < 1:
        raise ValueError("Процент должен быть числом между 0 и 1, не включая 0 и 1.")

    if not isinstance(column, str):
        raise ValueError("Столбец должен быть строкой (имя).")

    if column not in df.columns:
        raise ValueError(f"Столбец '{column}' не найден в датафрейме.")

    # Создаем копию DataFrame
    df_copy = df.copy()

    n_rows = df_copy.shape[0]
    n_remove = int(round(n_rows * percent))

    indices_to_remove = np.random.choice(df_copy.index, size=n_remove, replace=False)

    df_copy.loc[indices_to_remove, column] = np.nan

    return df_copy


def remove_contiguous_values(df: pd.DataFrame, column: str, percent: float) -> pd.DataFrame:
    """
    Создает новый DataFrame, в котором удален непрерывный блок значений в указанном столбце,
    составляющий заданный процент от общего числа строк. Исходный DataFrame остается неизменным.

    :param df: Исходный pandas DataFrame
    :param column: Название столбца (str)
    :param percent: Доля значений для удаления (float, 0 < percent < 1)
    :return: Новый DataFrame с удаленным блоком значений
    :raises ValueError: Если percent не в диапазоне (0, 1), column не строка или не найден в DataFrame
    """
    if not 0 < percent < 1:
        raise ValueError("Процент должен быть числом между 0 и 1, не включая 0 и 1.")
    if not isinstance(column, str):
        raise ValueError("Столбец должен быть строкой (имя).")
    if column not in df.columns:
        raise ValueError(f"Столбец '{column}' не найден в датафрейме.")

    df_copy = df.copy()
    n_rows = df_copy.shape[0]
    n_remove = int(round(n_rows * percent))

    if n_remove > 0:
        start = np.random.randint(0, n_rows - n_remove + 1)
        df_copy.iloc[start : start + n_remove, df_copy.columns.get_loc(column)] = np.nan

    return df_copy


def remove_rows_with_missing_values(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Удаляет строки с пропущенными значениями в указанном столбце.

    :param df: Исходный pandas DataFrame
    :param column: Название столбца, по которому удаляются строки с NaN
    :return: Модифицированный DataFrame без строк с пропусками в указанном столбце
    :raises ValueError: Если column не является строкой или не существует в df
    """
    if not isinstance(column, str):
        raise ValueError("Столбец должен быть строкой (имя столбца).")

    if column not in df.columns:
        raise ValueError(f"Столбец '{column}' не найден в датафрейме.")

    df_cleaned = df.dropna(subset=[column])
    return df_cleaned


def fill_missing_with_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Заполняет пропущенные значения в указанном столбце средним значением этого столбца.

    :param df: Исходный pandas DataFrame
    :param column: Название столбца, в котором нужно заполнить пропуски
    :return: Модифицированный DataFrame с заполненными пропусками
    :raises ValueError: Если column не является строкой, не существует в df или не является числовым
    """
    if not isinstance(column, str):
        raise ValueError("Столбец должен быть строкой (имя столбца).")

    if column not in df.columns:
        raise ValueError(f"Столбец '{column}' не найден в датафрейме.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Столбец '{column}' должен быть числовым для вычисления среднего.")

    mean_value = df[column].mean()
    df[column] = df[column].fillna(mean_value)
    return df


def fill_missing_with_regression(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Заполняет пропущенные значения в указанном числовом столбце датафрейма с помощью линейной регрессии.
    Выполняет полную предобработку данных, включая обработку даты, кодирование категориальных признаков и масштабирование.

    :param df: Исходный pandas DataFrame с пропусками в столбце column
    :param column: Название столбца, в котором нужно заполнить пропуски (числовой столбец)
    :return: Модифицированный DataFrame с заполненными пропусками
    :raises ValueError: Если column не является строкой, не существует в df или не является допустимым числовым столбцом
    """
    if not isinstance(column, str) or column not in df.columns:
        raise ValueError("column должен быть строкой и существовать в DataFrame")
    valid_columns = ["Долгота", "Широта", "Количество товаров", "Стоимость"]
    if column not in valid_columns:
        raise ValueError("column должен быть одним из: " + ", ".join(valid_columns))

    print(df.isnull().sum())

    df_copy = df.copy()

    if df_copy[column].isna().sum() == 0:
        return df_copy

    if "Дата и время" in df_copy.columns:
        df_copy["Год"] = pd.to_datetime(df_copy["Дата и время"]).dt.year
        df_copy["Месяц"] = pd.to_datetime(df_copy["Дата и время"]).dt.month
        df_copy["День"] = pd.to_datetime(df_copy["Дата и время"]).dt.day
        df_copy = df_copy.drop(columns=["Дата и время"])

    numeric_columns = ["Долгота", "Широта", "Количество товаров", "Стоимость"]
    categorical_columns = ["Название магазина", "Категория", "Бренд", "Номер карты"]
    numeric_columns = [col for col in numeric_columns if col in df_copy.columns and col != column]
    categorical_columns = [col for col in categorical_columns if col in df_copy.columns]

    train_df = df_copy[df_copy[column].notna()].copy()
    test_df = df_copy[df_copy[column].isna()].copy()

    X_train = train_df.drop(columns=[column])
    y_train = train_df[column]
    X_test = test_df.drop(columns=[column])

    num_imputer = SimpleImputer(strategy="mean")
    if numeric_columns:
        X_train[numeric_columns] = num_imputer.fit_transform(X_train[numeric_columns])
        X_test[numeric_columns] = num_imputer.transform(X_test[numeric_columns])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    if categorical_columns:
        X_train[categorical_columns] = cat_imputer.fit_transform(X_train[categorical_columns])
        X_test[categorical_columns] = cat_imputer.transform(X_test[categorical_columns])

    ohe_columns = [col for col in ["Название магазина", "Категория"] if col in categorical_columns]
    if ohe_columns:
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        ohe_train = pd.DataFrame(ohe.fit_transform(X_train[ohe_columns]), columns=ohe.get_feature_names_out(), index=X_train.index)
        ohe_test = pd.DataFrame(ohe.transform(X_test[ohe_columns]), columns=ohe.get_feature_names_out(), index=X_test.index)
        X_train = pd.concat([X_train.drop(columns=ohe_columns), ohe_train], axis=1)
        X_test = pd.concat([X_test.drop(columns=ohe_columns), ohe_test], axis=1)

    freq_columns = [col for col in ["Бренд", "Номер карты"] if col in categorical_columns]
    for col in freq_columns:
        freq_encoding = X_train[col].value_counts(normalize=True)
        X_train[col] = X_train[col].map(freq_encoding).fillna(0)
        X_test[col] = X_test[col].map(freq_encoding).fillna(0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ridge = Ridge(random_state=42)
    param_grid = {"alpha": [0.1, 1.0, 10.0, 100.0]}
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring="neg_mean_squared_error")
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    best_model.fit(X_train_scaled, y_train)

    y_pred = best_model.predict(X_test_scaled)
    df_copy.loc[df_copy[column].isna(), column] = y_pred

    return df_copy
