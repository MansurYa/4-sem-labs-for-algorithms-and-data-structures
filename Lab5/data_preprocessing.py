import pandas as pd
import numpy as np
import json
import hashlib
import pickle
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import silhouette_score
import warnings

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Константы
RANDOM_STATE = 42

class DataType(Enum):
    """Типы данных для колонок."""
    IDENTIFIER = "identifier"
    NUMERICAL_CONTINUOUS = "numerical_continuous"
    NUMERICAL_DISCRETE = "numerical_discrete"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    CATEGORICAL_ORDINAL = "categorical_ordinal"

class ScalingMethod(Enum):
    """Методы нормализации числовых данных."""
    STANDARD = "standard"
    MINMAX = "minmax"
    NONE = "none"

class EncodingMethod(Enum):
    """Методы кодирования категориальных данных."""
    LABEL = "label"
    ONEHOT = "onehot"
    ORDINAL = "ordinal"

@dataclass
class ColumnConfig:
    """Конфигурация обработки колонки."""
    name: str
    data_type: DataType
    scaling_method: Optional[ScalingMethod] = None
    encoding_method: Optional[EncodingMethod] = None
    ordinal_categories: Optional[List[str]] = None
    is_quasi_identifier: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColumnConfig':
        """
        Создание из словаря с обработкой ошибок.
        """
        try:
            try:
                data_type = DataType(data['data_type'])
            except (ValueError, KeyError):
                logger.warning(f"Неизвестный тип данных: {data.get('data_type')}. Используется CATEGORICAL_NOMINAL")
                data_type = DataType.CATEGORICAL_NOMINAL

            scaling_method = None
            if data.get('scaling_method'):
                try:
                    scaling_method = ScalingMethod(data['scaling_method'])
                except ValueError:
                    logger.warning(f"Неизвестный метод нормализации: {data.get('scaling_method')}. Используется STANDARD")
                    scaling_method = ScalingMethod.STANDARD

            encoding_method = None
            if data.get('encoding_method'):
                try:
                    encoding_method = EncodingMethod(data['encoding_method'])
                except ValueError:
                    logger.warning(f"Неизвестный метод кодирования: {data.get('encoding_method')}. Используется LABEL")
                    encoding_method = EncodingMethod.LABEL

            return cls(
                name=data.get('name', 'unknown'),
                data_type=data_type,
                scaling_method=scaling_method,
                encoding_method=encoding_method,
                ordinal_categories=data.get('ordinal_categories'),
                is_quasi_identifier=data.get('is_quasi_identifier', False)
            )

        except Exception as e:
            logger.error(f"Критическая ошибка при десериализации конфигурации колонки: {e}")

            # ИСПРАВЛЕНИЕ: логически совместимая резервная конфигурация
            return cls(
                name=data.get('name', 'unknown'),
                data_type=DataType.CATEGORICAL_NOMINAL,
                scaling_method=None,  # Категориальные данные не нормализуются
                encoding_method=EncodingMethod.LABEL,
                ordinal_categories=None,
                is_quasi_identifier=False
            )

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        result = {
            'name': self.name,
            'data_type': self.data_type.value,
            'is_quasi_identifier': self.is_quasi_identifier
        }

        if self.scaling_method:
            result['scaling_method'] = self.scaling_method.value

        if self.encoding_method:
            result['encoding_method'] = self.encoding_method.value

        if self.ordinal_categories:
            result['ordinal_categories'] = self.ordinal_categories

        return result

@dataclass
class PreprocessingConfig:
    """Конфигурация предобработки данных."""
    columns: Dict[str, ColumnConfig]
    random_state: int = RANDOM_STATE
    cache_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            'columns': {name: config.to_dict() for name, config in self.columns.items()},
            'random_state': self.random_state,
            'cache_enabled': self.cache_enabled
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreprocessingConfig':
        """Создание из словаря."""
        columns = {}
        for name, config_data in data.get('columns', {}).items():
            columns[name] = ColumnConfig.from_dict(config_data)

        return cls(
            columns=columns,
            random_state=data.get('random_state', RANDOM_STATE),
            cache_enabled=data.get('cache_enabled', True)
        )

    def save_to_file(self, filepath: str) -> None:
        """Сохранение конфигурации в JSON файл."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Конфигурация сохранена в {filepath}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении конфигурации: {e}")
            raise

    @classmethod
    def load_from_file(cls, filepath: str) -> 'PreprocessingConfig':
        """Загрузка конфигурации из JSON файла."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Конфигурация загружена из {filepath}")
            return cls.from_dict(data)
        except FileNotFoundError:
            logger.error(f"Файл конфигурации не найден: {filepath}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON в файле {filepath}: {e}")
            raise ValueError(f"Поврежденный файл конфигурации: {e}")
        except Exception as e:
            logger.error(f"Неожиданная ошибка при загрузке конфигурации: {e}")
            raise

class DataLoader:
    """Класс для загрузки и анализа CSV данных."""

    def __init__(self, filepath: str):
        """
        :param filepath: Путь к CSV файлу
        """
        self.filepath = filepath
        self.data = None
        self.analysis_results = None

        if not Path(filepath).exists():
            raise FileNotFoundError(f"Файл не найден: {filepath}")

    def load_data(self) -> pd.DataFrame:
        """
        Загрузка данных из CSV файла.

        :return: Загруженные данные
        """
        try:
            self.data = pd.read_csv(self.filepath, encoding='utf-8')
            logger.info(f"Данные загружены: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {self.filepath}: {e}")
            raise

    def analyze_column_types(self) -> Dict[str, Dict[str, Any]]:
        """
        Анализ типов колонок и их характеристик.

        :return: Словарь с анализом каждой колонки
        """
        if self.data is None:
            raise ValueError("Данные не загружены. Вызовите load_data() сначала.")

        results = {}

        for column in self.data.columns:
            column_stats = self._analyze_single_column(column)
            results[column] = column_stats

        self.analysis_results = results
        return results

    def _analyze_single_column(self, column: str) -> Dict[str, Any]:
        """
        Анализ одной колонки.

        :param column: Название колонки
        :return: Статистика колонки
        """
        series = self.data[column]

        stats = {
            'dtype': str(series.dtype),
            'unique_count': series.nunique(),
            'unique_percentage': (series.nunique() / len(series)) * 100,
            'missing_count': series.isnull().sum(),
            'missing_percentage': (series.isnull().sum() / len(series)) * 100
        }

        # Статистика для числовых типов
        if pd.api.types.is_numeric_dtype(series):
            non_null_series = series.dropna()
            if len(non_null_series) > 0:
                stats.update({
                    'min_value': float(non_null_series.min()),
                    'max_value': float(non_null_series.max()),
                    'mean_value': float(non_null_series.mean()),
                    'std_value': float(non_null_series.std()),
                    'median_value': float(non_null_series.median())
                })

        # Статистика для категориальных типов
        elif pd.api.types.is_object_dtype(series):
            value_counts = series.value_counts()
            if len(value_counts) > 0:
                stats.update({
                    'most_frequent_value': str(value_counts.index[0]),
                    'most_frequent_count': int(value_counts.iloc[0]),
                    'categories': list(value_counts.index[:10])  # Топ-10 категорий
                })

        # Определение типа данных
        stats['classified_type'] = self._classify_column_type(column, stats)

        return stats

    def _classify_column_type(self, column: str, stats: Dict[str, Any]) -> DataType:
        """
        Классификация типа колонки с защитой от численной нестабильности.
        """
        # Правило 1: Идентификаторы
        if column.lower() in ['id', 'student_id', 'user_id']:
            return DataType.IDENTIFIER

        # Правило 2: Числовые типы
        if 'int' in stats['dtype'] or 'float' in stats['dtype']:
            unique_percentage = stats['unique_percentage']
            unique_count = stats['unique_count']

            if unique_percentage > 50 and unique_count > 20:
                return DataType.NUMERICAL_CONTINUOUS

            if unique_count <= 20:
                return DataType.NUMERICAL_DISCRETE

            # ИСПРАВЛЕНИЕ: Защита от численной нестабильности
            if 'std_value' in stats and stats['std_value'] > 0:
                mean_value = stats.get('mean_value', 0)

                if abs(mean_value) > 1e-6:
                    cv = stats['std_value'] / abs(mean_value)
                    if cv > 0.3:
                        return DataType.NUMERICAL_CONTINUOUS
                    else:
                        return DataType.NUMERICAL_DISCRETE
                else:
                    if stats['std_value'] >= 1.0:  # ИСПРАВЛЕНИЕ: >= вместо >
                        return DataType.NUMERICAL_CONTINUOUS
                    else:
                        return DataType.NUMERICAL_DISCRETE

            return DataType.NUMERICAL_CONTINUOUS

        # Правило 3: Категориальные типы
        if pd.api.types.is_object_dtype(self.data[column]):
            unique_count = stats['unique_count']

            # Проверка на ординальные категории
            if self._is_ordinal_column(column, stats):
                return DataType.CATEGORICAL_ORDINAL
            else:
                return DataType.CATEGORICAL_NOMINAL

        return DataType.CATEGORICAL_NOMINAL

    def _is_ordinal_column(self, column: str, stats: Dict[str, Any]) -> bool:
        """
        Определение ординальности категориальной колонки.
        """
        ordinal_indicators = [
            'quality', 'level', 'grade', 'rating', 'education'
        ]

        if any(indicator in column.lower() for indicator in ordinal_indicators):
            return True

        if 'categories' in stats:
            categories = stats['categories']
            ordinal_patterns = [
                ['poor', 'fair', 'good'],
                ['low', 'medium', 'high'],
                ['bad', 'average', 'good'],
                ['none', 'high school', 'bachelor', 'master']
            ]

            for pattern in ordinal_patterns:
                if any(cat.lower() in [p.lower() for p in pattern] for cat in categories):
                    return True

        return False

    def get_default_config(self) -> PreprocessingConfig:
        """
        Генерация конфигурации по умолчанию на основе анализа данных.

        :return: Конфигурация предобработки
        """
        if self.analysis_results is None:
            self.analyze_column_types()

        columns_config = {}

        for column_name, stats in self.analysis_results.items():
            data_type = stats['classified_type']

            # Определение методов обработки
            scaling_method = None
            encoding_method = None
            ordinal_categories = None

            if data_type in [DataType.NUMERICAL_CONTINUOUS, DataType.NUMERICAL_DISCRETE]:
                scaling_method = ScalingMethod.STANDARD
            elif data_type == DataType.CATEGORICAL_NOMINAL:
                encoding_method = EncodingMethod.ONEHOT if stats['unique_count'] <= 10 else EncodingMethod.LABEL
            elif data_type == DataType.CATEGORICAL_ORDINAL:
                encoding_method = EncodingMethod.ORDINAL
                if 'categories' in stats:
                    ordinal_categories = self._get_ordinal_order(stats['categories'])

            # Определение квази-идентификаторов для student_habits_performance.csv
            is_quasi_identifier = column_name.lower() in [
                'age', 'gender', 'parental_education_level',
                'internet_quality', 'part_time_job'
            ]

            columns_config[column_name] = ColumnConfig(
                name=column_name,
                data_type=data_type,
                scaling_method=scaling_method,
                encoding_method=encoding_method,
                ordinal_categories=ordinal_categories,
                is_quasi_identifier=is_quasi_identifier
            )

        return PreprocessingConfig(columns=columns_config)

    def _get_ordinal_order(self, categories: List[str]) -> List[str]:
        """
        Определение порядка ординальных категорий.
        """
        ordinal_mappings = {
            ('poor', 'fair', 'good'): ['Poor', 'Fair', 'Good'],
            ('none', 'high school', 'bachelor', 'master'): ['None', 'High School', 'Bachelor', 'Master'],
            ('poor', 'average', 'good'): ['Poor', 'Average', 'Good']
        }

        categories_lower = [cat.lower() for cat in categories]

        for pattern, order in ordinal_mappings.items():
            if all(p in categories_lower for p in pattern):
                return [cat for cat in order if cat in categories]

        return sorted(categories)

class DataPreprocessor:
    """Класс для предобработки данных согласно конфигурации."""

    def __init__(self, config: PreprocessingConfig):
        """
        :param config: Конфигурация предобработки
        """
        self.config = config
        self.fitted_transformers: Dict[str, Any] = {}
        self.feature_names_out_: Optional[List[str]] = None
        self.is_fitted = False

        np.random.seed(config.random_state)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обучение трансформеров и применение предобработки.

        :param data: Исходные данные
        :return: Предобработанные данные
        """
        self.fit(data)
        return self.transform(data)

    def fit(self, data: pd.DataFrame) -> 'DataPreprocessor':
        """
        Обучение трансформеров на данных.

        :param data: Данные для обучения
        :return: self
        """
        logger.info("Начинается обучение предобработчика...")

        # Валидация данных
        self._validate_data(data)

        # Обучение трансформеров по типам данных
        numerical_columns = self._get_columns_by_type(data, [DataType.NUMERICAL_CONTINUOUS, DataType.NUMERICAL_DISCRETE])
        categorical_columns = self._get_columns_by_type(data, [DataType.CATEGORICAL_NOMINAL, DataType.CATEGORICAL_ORDINAL])

        if numerical_columns:
            self._fit_numerical_transformers(data, numerical_columns)

        if categorical_columns:
            self._fit_categorical_transformers(data, categorical_columns)

        self.is_fitted = True
        logger.info("Обучение предобработчика завершено")

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применение предобработки к данным.

        :param data: Данные для трансформации
        :return: Трансформированные данные
        """
        if not self.is_fitted:
            raise ValueError("Предобработчик не обучен. Вызовите fit() сначала.")

        logger.info("Применение предобработки к данным...")

        self._validate_data(data)

        # Разделение по типам данных
        numerical_columns = self._get_columns_by_type(data, [DataType.NUMERICAL_CONTINUOUS, DataType.NUMERICAL_DISCRETE])
        categorical_columns = self._get_columns_by_type(data, [DataType.CATEGORICAL_NOMINAL, DataType.CATEGORICAL_ORDINAL])
        identifier_columns = self._get_columns_by_type(data, [DataType.IDENTIFIER])

        result_parts = []

        # Обработка числовых колонок
        if numerical_columns:
            numerical_result = self._transform_numerical_columns(data, numerical_columns)
            if not numerical_result.empty:
                result_parts.append(numerical_result)

        # Обработка категориальных колонок
        if categorical_columns:
            categorical_result = self._transform_categorical_columns(data, categorical_columns)
            if not categorical_result.empty:
                result_parts.append(categorical_result)

        # Объединение результатов
        if result_parts:
            result = pd.concat(result_parts, axis=1)
        else:
            result = pd.DataFrame(index=data.index)

        # Сохранение имен признаков
        self.feature_names_out_ = list(result.columns)

        logger.info(f"Предобработка завершена: {data.shape} → {result.shape}")

        return result

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Валидация входных данных."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Данные должны быть pandas DataFrame")

        if data.empty:
            raise ValueError("Данные не могут быть пустыми")

        # Проверка соответствия колонок конфигурации
        missing_columns = set(self.config.columns.keys()) - set(data.columns)
        if missing_columns:
            logger.warning(f"Отсутствующие колонки в данных: {missing_columns}")

    def _get_columns_by_type(self, data: pd.DataFrame, data_types: List[DataType]) -> List[str]:
        """Получение колонок определенных типов."""
        columns = []
        for column_name in data.columns:
            if column_name in self.config.columns:
                column_config = self.config.columns[column_name]
                if column_config.data_type in data_types:
                    columns.append(column_name)
        return columns

    def _fit_numerical_transformers(self, data: pd.DataFrame, columns: List[str]) -> None:
        """Обучение трансформеров для числовых данных."""
        for column in columns:
            config = self.config.columns[column]

            if config.scaling_method == ScalingMethod.STANDARD:
                scaler = StandardScaler()
                column_data = data[column].values.reshape(-1, 1)
                scaler.fit(column_data)
                self.fitted_transformers[f"{column}_scaler"] = scaler

            elif config.scaling_method == ScalingMethod.MINMAX:
                scaler = MinMaxScaler()
                column_data = data[column].values.reshape(-1, 1)
                scaler.fit(column_data)
                self.fitted_transformers[f"{column}_scaler"] = scaler

    def _fit_categorical_transformers(self, data: pd.DataFrame, columns: List[str]) -> None:
        """Обучение трансформеров для категориальных данных."""
        for column in columns:
            config = self.config.columns[column]

            # Детерминированное заполнение пропусков
            clean_data = self._fill_missing_values_deterministic(data[column], column)

            if config.encoding_method == EncodingMethod.LABEL:
                encoder = LabelEncoder()
                encoder.fit(clean_data)
                self.fitted_transformers[f"{column}_encoder"] = encoder

            elif config.encoding_method == EncodingMethod.ONEHOT:
                encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                encoder.fit(clean_data.values.reshape(-1, 1))
                self.fitted_transformers[f"{column}_encoder"] = encoder

            elif config.encoding_method == EncodingMethod.ORDINAL:
                if config.ordinal_categories:
                    ordinal_mapping = {cat: i for i, cat in enumerate(config.ordinal_categories)}
                    self.fitted_transformers[f"{column}_ordinal_mapping"] = ordinal_mapping

    def _fill_missing_values_deterministic(self, series: pd.Series, column_name: str) -> pd.Series:
        """
        Детерминированное заполнение пропущенных значений.

        :param series: Серия данных с возможными пропусками
        :param column_name: Имя колонки для логирования
        :return: серия с заполненными пропусками
        """
        if series.isnull().sum() == 0:
            return series

        value_counts = series.value_counts()

        if len(value_counts) == 0:
            fill_value = 'Unknown'
            logger.debug(f"Колонка {column_name}: все значения NaN, заполняем 'Unknown'")
        else:
            max_count = value_counts.iloc[0]
            modal_values = value_counts[value_counts == max_count].index.tolist()

            if len(modal_values) == 1:
                fill_value = modal_values[0]
                logger.debug(f"Колонка {column_name}: заполняем единственным модальным значением '{fill_value}'")
            else:
                # ИСПРАВЛЕНИЕ: Детерминированный выбор при равных частотах
                fill_value = sorted(modal_values)[0]
                logger.debug(f"Колонка {column_name}: множественные модальные значения {modal_values}, выбираем '{fill_value}'")

        return series.fillna(fill_value)

    def _transform_numerical_columns(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Трансформация числовых колонок."""
        result_dataframes = []

        for column in columns:
            config = self.config.columns[column]

            if config.scaling_method == ScalingMethod.NONE:
                result_dataframes.append(pd.DataFrame({column: data[column]}, index=data.index))
                continue

            scaler_key = f"{column}_scaler"
            if scaler_key in self.fitted_transformers:
                scaler = self.fitted_transformers[scaler_key]
                column_data = data[column].values.reshape(-1, 1)
                scaled_data = scaler.transform(column_data).flatten()
                result_dataframes.append(pd.DataFrame({column: scaled_data}, index=data.index))
            else:
                result_dataframes.append(pd.DataFrame({column: data[column]}, index=data.index))

        return pd.concat(result_dataframes, axis=1) if result_dataframes else pd.DataFrame(index=data.index)

    def _transform_categorical_columns(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Трансформация категориальных колонок."""
        result_dataframes = []

        for column in columns:
            config = self.config.columns[column]

            column_data = self._fill_missing_values_deterministic(data[column], column)

            if config.encoding_method == EncodingMethod.LABEL:
                encoder_key = f"{column}_encoder"
                if encoder_key in self.fitted_transformers:
                    encoder = self.fitted_transformers[encoder_key]
                    try:
                        encoded_data = encoder.transform(column_data)
                        result_dataframes.append(pd.DataFrame({column: encoded_data}, index=data.index))
                    except ValueError:
                        # Обработка неизвестных категорий
                        known_categories = set(encoder.classes_)
                        column_data_fixed = column_data.apply(
                            lambda x: x if x in known_categories else encoder.classes_[0]
                        )
                        encoded_data = encoder.transform(column_data_fixed)
                        result_dataframes.append(pd.DataFrame({column: encoded_data}, index=data.index))

            elif config.encoding_method == EncodingMethod.ONEHOT:
                encoder_key = f"{column}_encoder"
                if encoder_key in self.fitted_transformers:
                    encoder = self.fitted_transformers[encoder_key]
                    encoded_data = encoder.transform(column_data.values.reshape(-1, 1))

                    feature_names = [f"{column}_{cat}" for cat in encoder.categories_[0][1:]]  # drop='first'
                    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=data.index)
                    result_dataframes.append(encoded_df)

            elif config.encoding_method == EncodingMethod.ORDINAL:
                mapping_key = f"{column}_ordinal_mapping"
                if mapping_key in self.fitted_transformers:
                    ordinal_mapping = self.fitted_transformers[mapping_key]
                    encoded_data = column_data.map(ordinal_mapping)
                    encoded_data = encoded_data.fillna(0)  # Неизвестные категории -> 0
                    result_dataframes.append(pd.DataFrame({column: encoded_data}, index=data.index))

        return pd.concat(result_dataframes, axis=1) if result_dataframes else pd.DataFrame(index=data.index)

    def get_feature_names_out(self) -> List[str]:
        """Получение имен выходных признаков."""
        if self.feature_names_out_ is None:
            raise ValueError("Трансформер не применен к данным")
        return self.feature_names_out_

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Получение сводки о выполненной предобработке."""
        summary = {
            'is_fitted': self.is_fitted,
            'random_state': self.config.random_state,
            'transformers_count': len(self.fitted_transformers),
            'output_features_count': len(self.feature_names_out_) if self.feature_names_out_ else 0
        }

        if self.feature_names_out_:
            summary['output_features'] = self.feature_names_out_

        return summary

class DataCacheManager:
    """Менеджер кэширования предобработанных данных."""

    def __init__(self, cache_dir: str = "data_cache"):
        """
        :param cache_dir: Директория для кэширования
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.data_path = self.cache_dir / "processed_data.pkl"
        self.preprocessor_path = self.cache_dir / "preprocessor.pkl"
        self.config_path = self.cache_dir / "config.json"
        self.metadata_path = self.cache_dir / "metadata.json"

    def is_cache_valid(self, source_filepath: str, config: PreprocessingConfig) -> bool:
        """
        Проверка актуальности кэша.

        :param source_filepath: Путь к исходному файлу данных
        :param config: Текущая конфигурация предобработки
        :return: True если кэш актуален
        """
        try:
            if not all(path.exists() for path in [self.data_path, self.preprocessor_path,
                                                  self.config_path, self.metadata_path]):
                return False

            # Проверка метаданных
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Проверка хэша исходного файла
            current_hash = self._compute_file_hash(source_filepath)
            if metadata.get('source_file_hash') != current_hash:
                logger.debug("Исходный файл изменился")
                return False

            # ИСПРАВЛЕНИЕ: Надежное сравнение конфигурации
            try:
                cached_config = PreprocessingConfig.load_from_file(self.config_path)

                current_config_json = json.dumps(config.to_dict(), sort_keys=True, ensure_ascii=False)
                cached_config_json = json.dumps(cached_config.to_dict(), sort_keys=True, ensure_ascii=False)

                if current_config_json != cached_config_json:
                    logger.debug("Конфигурация предобработки изменилась")
                    return False

            except Exception as e:
                logger.debug(f"Ошибка при сравнении конфигурации: {e}")
                return False

            logger.info("Кэш актуален и может быть использован")
            return True

        except Exception as e:
            logger.debug(f"Ошибка при проверке кэша: {e}")
            return False

    def save_to_cache(self, processed_data: pd.DataFrame,
                     preprocessor: DataPreprocessor,
                     source_filepath: str,
                     config: PreprocessingConfig) -> None:
        """
        Сохранение данных в кэш.
        """
        try:
            # Сохранение обработанных данных
            with open(self.data_path, 'wb') as f:
                pickle.dump(processed_data, f)

            # Сохранение предобработчика
            with open(self.preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor, f)

            # Сохранение конфигурации
            config.save_to_file(str(self.config_path))

            # Сохранение метаданных
            metadata = {
                'source_file_hash': self._compute_file_hash(source_filepath),
                'cache_created_at': datetime.now().isoformat(),
                'data_shape': list(processed_data.shape),
                'config_hash': self._compute_config_hash(config)
            }

            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info("Данные сохранены в кэш")

        except Exception as e:
            logger.error(f"Ошибка при сохранении в кэш: {e}")
            raise

    def load_from_cache(self) -> Tuple[pd.DataFrame, DataPreprocessor, PreprocessingConfig]:
        """
        Загрузка данных из кэша.

        :return: (processed_data, preprocessor, config)
        """
        try:
            # Загрузка обработанных данных
            with open(self.data_path, 'rb') as f:
                processed_data = pickle.load(f)

            # Загрузка предобработчика
            with open(self.preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)

            # Загрузка конфигурации
            config = PreprocessingConfig.load_from_file(str(self.config_path))

            logger.info("Данные успешно загружены из кэша")

            return processed_data, preprocessor, config

        except Exception as e:
            logger.error(f"Ошибка при загрузке из кэша: {e}")
            raise

    def clear_cache(self) -> None:
        """Очистка кэша."""
        try:
            for path in [self.data_path, self.preprocessor_path, self.config_path, self.metadata_path]:
                if path.exists():
                    path.unlink()
            logger.info("Кэш очищен")
        except Exception as e:
            logger.error(f"Ошибка при очистке кэша: {e}")
            raise

    def get_cache_info(self) -> Dict[str, Any]:
        """Получение информации о кэше."""
        info = {
            'cache_dir': str(self.cache_dir),
            'cache_exists': all(path.exists() for path in [self.data_path, self.preprocessor_path])
        }

        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                info.update(metadata)
            except Exception as e:
                info['metadata_error'] = str(e)

        return info

    def _compute_file_hash(self, filepath: str) -> str:
        """Вычисление MD5 хэша файла."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _compute_config_hash(self, config: PreprocessingConfig) -> str:
        """
        Вычисление надежного хэша конфигурации.
        """
        config_json = json.dumps(config.to_dict(), sort_keys=True, ensure_ascii=False)
        config_hash = hashlib.md5(config_json.encode('utf-8')).hexdigest()
        return config_hash

class KAnonymityAnalyzer:
    """Анализатор K-анонимности данных."""

    def __init__(self):
        """Инициализация анализатора."""
        self.analysis_results: Optional[Dict[str, Any]] = None
        self.equivalence_classes: Optional[pd.DataFrame] = None

    def analyze_k_anonymity(self, data: pd.DataFrame,
                           quasi_identifiers: List[str]) -> Dict[str, Any]:
        """
        Анализ K-анонимности данных.

        :param data: Данные для анализа
        :param quasi_identifiers: Список квази-идентификаторов
        :return: Результаты анализа K-анонимности
        """
        logger.info(f"Начинается анализ K-анонимности с QI: {quasi_identifiers}")

        # Валидация входных данных
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Данные должны быть pandas DataFrame")

        if data.empty:
            raise ValueError("Данные не могут быть пустыми")

        missing_columns = set(quasi_identifiers) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Отсутствующие квази-идентификаторы в данных: {missing_columns}")

        # Вычисление групп эквивалентности
        equivalence_classes = self._compute_equivalence_classes(data, quasi_identifiers)

        # Вычисление K-анонимности
        k_anonymity_value = equivalence_classes['group_size'].min()

        # Детальный анализ групп
        groups_analysis = self._analyze_groups(equivalence_classes)

        # Анализ уязвимостей
        vulnerability_analysis = self._analyze_vulnerabilities(data, equivalence_classes, quasi_identifiers)

        # Генерация рекомендаций
        recommendations = self._generate_recommendations(k_anonymity_value, groups_analysis)

        # Сборка результатов
        results = {
            'k_anonymity_value': int(k_anonymity_value),
            'total_records': len(data),
            'equivalence_classes_count': len(equivalence_classes),
            'quasi_identifiers': quasi_identifiers,
            'groups_analysis': groups_analysis,
            'vulnerability_analysis': vulnerability_analysis,
            'recommendations': recommendations,
            'analysis_timestamp': datetime.now().isoformat()
        }

        self.analysis_results = results
        self.equivalence_classes = equivalence_classes

        logger.info(f"Анализ K-анонимности завершен. K = {k_anonymity_value}")

        return results

    def _compute_equivalence_classes(self, data: pd.DataFrame,
                                   quasi_identifiers: List[str]) -> pd.DataFrame:
        """
        Вычисление групп эквивалентности.

        :param data: Исходные данные
        :param quasi_identifiers: Квази-идентификаторы
        :return: DataFrame с группами эквивалентности
        """
        # Группировка по квази-идентификаторам
        grouped = data.groupby(quasi_identifiers).size().reset_index(name='group_size')

        # Добавление дополнительной информации
        grouped = grouped.sort_values('group_size', ascending=True)
        grouped['group_id'] = range(len(grouped))

        return grouped

    def _analyze_groups(self, equivalence_classes: pd.DataFrame) -> Dict[str, Any]:
        """Анализ статистики групп эквивалентности."""
        group_sizes = equivalence_classes['group_size']

        analysis = {
            'total_groups': len(equivalence_classes),
            'min_group_size': int(group_sizes.min()),
            'max_group_size': int(group_sizes.max()),
            'mean_group_size': float(group_sizes.mean()),
            'median_group_size': float(group_sizes.median()),
            'std_group_size': float(group_sizes.std()),
            'vulnerable_groups_count': int((group_sizes == 1).sum()),
            'vulnerable_records_count': int(equivalence_classes[equivalence_classes['group_size'] == 1]['group_size'].sum()),
            'size_distribution': group_sizes.value_counts().to_dict()
        }

        return analysis

    def _analyze_vulnerabilities(self, data: pd.DataFrame,
                               equivalence_classes: pd.DataFrame,
                               quasi_identifiers: List[str]) -> Dict[str, Any]:
        """Анализ уязвимостей приватности."""
        analysis = {}

        # Анализ уникальных записей
        unique_records = equivalence_classes[equivalence_classes['group_size'] == 1]
        analysis['unique_records'] = {
            'count': len(unique_records),
            'percentage': (len(unique_records) / len(equivalence_classes)) * 100 if len(equivalence_classes) > 0 else 0
        }

        # Анализ малых групп (размер < 5)
        small_groups = equivalence_classes[equivalence_classes['group_size'] < 5]
        analysis['small_groups'] = {
            'groups_count': len(small_groups),
            'affected_records': int(small_groups['group_size'].sum()),
            'percentage_of_groups': (len(small_groups) / len(equivalence_classes)) * 100 if len(equivalence_classes) > 0 else 0
        }

        # Анализ влияния каждого квази-идентификатора
        qi_impact = {}
        for qi in quasi_identifiers:
            if qi in data.columns:
                unique_values = data[qi].nunique()
                total_values = len(data)

                qi_impact[qi] = {
                    'unique_values_count': int(unique_values),
                    'uniqueness_ratio': unique_values / total_values if total_values > 0 else 0,
                    'entropy': float(self._calculate_entropy(data[qi]))
                }

        analysis['quasi_identifier_impact'] = qi_impact

        # Оценка риска де-анонимизации
        analysis['risk_assessment'] = self._assess_deanonymization_risk(equivalence_classes)

        return analysis

    def _calculate_entropy(self, series: pd.Series) -> float:
        """Вычисление энтропии для серии данных."""
        value_counts = series.value_counts()
        probabilities = value_counts / len(series)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Добавляем малое число для избежания log(0)
        return entropy

    def _assess_deanonymization_risk(self, equivalence_classes: pd.DataFrame) -> Dict[str, Any]:
        """
        Оценка риска де-анонимизации на основе распределения размеров групп.
        """
        group_sizes = equivalence_classes['group_size']
        total_records = group_sizes.sum()

        records_in_size_1 = equivalence_classes[equivalence_classes['group_size'] == 1]['group_size'].sum()
        records_in_size_2 = equivalence_classes[equivalence_classes['group_size'] == 2]['group_size'].sum()
        records_in_small_groups = equivalence_classes[equivalence_classes['group_size'] < 5]['group_size'].sum()

        percent_size_1 = (records_in_size_1 / total_records * 100) if total_records > 0 else 0
        percent_size_2 = (records_in_size_2 / total_records * 100) if total_records > 0 else 0
        percent_small = (records_in_small_groups / total_records * 100) if total_records > 0 else 0

        if percent_size_1 > 10 or percent_small > 20:
            risk_level = "HIGH"
            risk_description = "Высокий риск де-анонимизации из-за большого количества уникальных или малых групп"
        elif percent_size_1 > 5 or percent_small > 10:
            risk_level = "MEDIUM"
            risk_description = "Средний риск де-анонимизации"
        else:
            risk_level = "LOW"
            risk_description = "Низкий риск де-анонимизации"

        return {
            'risk_level': risk_level,
            'risk_description': risk_description,
            'metrics': {
                'records_in_unique_groups': int(records_in_size_1),
                'records_in_size_2_groups': int(records_in_size_2),
                'records_in_small_groups': int(records_in_small_groups),
                'percent_unique': float(percent_size_1),
                'percent_size_2': float(percent_size_2),
                'percent_small_groups': float(percent_small)
            }
        }

    def _generate_recommendations(self, k_value: int,
                                groups_analysis: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по улучшению K-анонимности."""
        recommendations = []

        if k_value == 1:
            recommendations.append("КРИТИЧНО: Присутствуют уникальные записи. Рассмотрите обобщение квази-идентификаторов.")
        elif k_value < 3:
            recommendations.append("ПРЕДУПРЕЖДЕНИЕ: Низкий уровень K-анонимности. Рекомендуется увеличить до K≥3.")
        elif k_value < 5:
            recommendations.append("Умеренный уровень анонимности. Для повышения безопасности рекомендуется K≥5.")
        else:
            recommendations.append("Хороший уровень K-анонимности для базовой защиты приватности.")

        vulnerable_percent = (groups_analysis['vulnerable_records_count'] /
                            (groups_analysis['mean_group_size'] * groups_analysis['total_groups']) * 100)

        if vulnerable_percent > 10:
            recommendations.append("Высокий процент уязвимых записей. Рассмотрите методы генерализации данных.")

        if groups_analysis['max_group_size'] / groups_analysis['mean_group_size'] > 10:
            recommendations.append("Неравномерное распределение размеров групп. Проверьте сбалансированность данных.")

        recommendations.extend([
            "Рассмотрите удаление или обобщение наиболее идентифицирующих атрибутов.",
            "Для критичных данных рекомендуется дополнительная защита (l-diversity, t-closeness).",
            "Регулярно пересматривайте список квази-идентификаторов при изменении данных."
        ])

        return recommendations

# ИСПРАВЛЕНИЕ: Интеграция с алгоритмом СПА
class FeatureSelector:
    """
    Класс для выбора наиболее информативных признаков с использованием алгоритма СПА.
    """

    def __init__(self, random_state: int = 42):
        """
        :param random_state: Seed для воспроизводимости
        """
        self.random_state = random_state
        self.selected_features_ = None
        self.selection_quality_ = None
        self.selection_history_ = None

        logger.info("Инициализирован FeatureSelector с интеграцией СПА")

    def select_features_spa(self, data: pd.DataFrame,
                           n_features: int,
                           clustering_method: str = 'single_linkage',
                           n_clusters: int = 3,
                           distance_metric: str = 'euclidean',
                           max_iterations: int = 1000) -> Tuple[List[str], float]:
        """
        Выбор признаков с использованием алгоритма СПА.

        :param data: Предобработанные данные для анализа
        :param n_features: Количество признаков для выбора
        :param clustering_method: Метод кластеризации для оценки качества
        :param n_clusters: Количество кластеров для оценки
        :param distance_metric: Метрика расстояния
        :param max_iterations: Максимальное количество итераций СПА
        :return: (выбранные признаки, качество)
        """
        try:
            # ИСПРАВЛЕНИЕ: Безопасный импорт математических алгоритмов
            from mathematical_algorithms import (
                single_linkage_clustering,
                spa_feature_selection
            )

            logger.info(f"Начинаем выбор {n_features} признаков из {data.shape[1]} доступных")

            def quality_function(feature_subset_data: np.ndarray) -> float:
                """функция оценки качества подмножества признаков."""
                try:
                    if feature_subset_data.shape[1] < 2:
                        return 0.0

                    # ИСПРАВЛЕНИЕ: Убираем некорректную проверку
                    # НЕПРАВИЛЬНО: if feature_subset_data.shape[0] < n_clusters: return 0.0

                    # ПРАВИЛЬНО: Проверяем минимальное количество объектов для кластеризации
                    if feature_subset_data.shape[0] < 2:
                        return 0.0

                    # НОВАЯ ПРОВЕРКА: Убеждаемся, что признаки не константные
                    for col_idx in range(feature_subset_data.shape[1]):
                        col_data = feature_subset_data[:, col_idx]
                        unique_count = len(np.unique(col_data))

                        # Если признак имеет слишком мало уникальных значений - штрафуем
                        if unique_count < 10:  # Минимум 10 уникальных значений
                            return 0.0

                        if np.std(col_data) < 1e-6:  # Почти константный столбец
                            return 0.0

                    # ПРОВЕРКА ОБЩЕГО РАЗНООБРАЗИЯ ДАННЫХ
                    total_variance = np.sum(np.var(feature_subset_data, axis=0))
                    if total_variance < 0.1:  # Слишком низкая общая дисперсия
                        return 0.0

                    labels, _ = single_linkage_clustering(
                        feature_subset_data,
                        n_clusters=min(n_clusters, feature_subset_data.shape[0] - 1),
                        metric=distance_metric,
                        verbose=False
                    )

                    sklearn_metric = 'euclidean' if distance_metric in ['euclidean_squared'] else distance_metric

                    if len(np.unique(labels)) > 1:
                        from sklearn.metrics import silhouette_score
                        try:
                            score = silhouette_score(feature_subset_data, labels, metric=sklearn_metric)
                            return max(0.0, score)
                        except Exception:
                            # Альтернативная оценка качества
                            return self._calculate_cluster_separation_ratio(feature_subset_data, labels)
                    else:
                        return 0.0

                except Exception as e:
                    return 0.0

            # Добавьте этот вспомогательный метод в класс FeatureSelector:
            def _calculate_cluster_separation_ratio(self, data, labels):
                """Альтернативная метрика: отношение межкластерного к внутрикластерному расстоянию."""
                try:
                    unique_labels = np.unique(labels)
                    if len(unique_labels) <= 1:
                        return 0.0

                    # Вычисляем центроиды кластеров
                    centroids = []
                    for label in unique_labels:
                        cluster_points = data[labels == label]
                        if len(cluster_points) > 0:
                            centroids.append(np.mean(cluster_points, axis=0))

                    centroids = np.array(centroids)

                    # Средние внутрикластерные расстояния
                    intra_cluster_distances = []
                    for label in unique_labels:
                        cluster_points = data[labels == label]
                        if len(cluster_points) > 1:
                            centroid = centroids[label]
                            distances = [np.linalg.norm(point - centroid) for point in cluster_points]
                            intra_cluster_distances.extend(distances)

                    # Межкластерные расстояния
                    inter_cluster_distances = []
                    for i in range(len(centroids)):
                        for j in range(i + 1, len(centroids)):
                            distance = np.linalg.norm(centroids[i] - centroids[j])
                            inter_cluster_distances.append(distance)

                    if len(intra_cluster_distances) == 0 or len(inter_cluster_distances) == 0:
                        return 0.0

                    avg_intra = np.mean(intra_cluster_distances)
                    avg_inter = np.mean(inter_cluster_distances)

                    if avg_intra > 0:
                        ratio = avg_inter / avg_intra
                        return min(1.0, ratio / 10.0)  # Нормализуем в [0, 1]
                    else:
                        return 1.0

                except Exception:
                    return 0.0

            selected_indices, best_quality, history = spa_feature_selection(
                data.values,
                n_features=n_features,
                quality_function=quality_function,
                max_iterations=max_iterations,
                random_state=self.random_state,
                verbose=True
            )

            selected_feature_names = [data.columns[i] for i in selected_indices]

            self.selected_features_ = selected_feature_names
            self.selection_quality_ = best_quality
            self.selection_history_ = history

            logger.info(f"СПА завершен. Выбранные признаки: {selected_feature_names}")
            logger.info(f"Качество выбора: {best_quality:.6f}")

            return selected_feature_names, best_quality

        except ImportError as e:
            logger.error(f"Не удалось импортировать математические алгоритмы: {e}")
            logger.info("Используется резервный метод выбора признаков")
            return self._fallback_feature_selection(data, n_features)

        except Exception as e:
            logger.error(f"Ошибка в алгоритме СПА: {e}")
            logger.info("Используется резервный метод выбора признаков")
            return self._fallback_feature_selection(data, n_features)

    def _fallback_feature_selection(self, data: pd.DataFrame, n_features: int) -> Tuple[List[str], float]:
        """Резервный метод выбора признаков на основе дисперсии."""
        logger.warning("Используется резервный выбор признаков по дисперсии")

        # ИСПРАВЛЕНИЕ: Обработка константных колонок
        feature_variances = data.var()
        feature_variances = feature_variances.fillna(0)  # NaN дисперсии заменяем на 0

        top_features = feature_variances.nlargest(n_features).index.tolist()

        # Если недостаточно признаков с ненулевой дисперсией
        if len(top_features) < n_features:
            remaining_features = [col for col in data.columns if col not in top_features]
            top_features.extend(remaining_features[:n_features - len(top_features)])

        quality_estimate = feature_variances.nlargest(n_features).mean() / (feature_variances.mean() + 1e-6)

        self.selected_features_ = top_features
        self.selection_quality_ = quality_estimate

        return top_features, quality_estimate

    def get_selection_summary(self) -> Dict[str, Any]:
        """
        :return: сводная информация
        """
        if self.selected_features_ is None:
            return {"status": "not_fitted"}

        return {
            "status": "fitted",
            "selected_features": self.selected_features_,
            "selection_quality": self.selection_quality_,
            "n_features_selected": len(self.selected_features_),
            "selection_method": "SPA" if self.selection_history_ else "variance"
        }


class StudentDataProcessor:
    """
    Главный класс для комплексной обработки данных student_habits_performance.csv.
    """

    def __init__(self, data_filepath: str, cache_dir: str = "data_cache"):
        """
        :param data_filepath: Путь к файлу student_habits_performance.csv
        :param cache_dir: Директория для кэширования результатов
        """
        self.data_filepath = data_filepath
        self.cache_dir = cache_dir

        self.data_loader = DataLoader(data_filepath)
        self.cache_manager = DataCacheManager(cache_dir)
        self.preprocessor = None
        self.k_anonymity_analyzer = KAnonymityAnalyzer()

        # ИСПРАВЛЕНИЕ: Добавление селектора признаков
        self.feature_selector = FeatureSelector(random_state=RANDOM_STATE)
        self.selected_features = None

        self.raw_data = None
        self.processed_data = None
        self.config = None
        self.processing_summary = {}

        logger.info(f"Инициализирован StudentDataProcessor для файла: {data_filepath}")

    def load_and_analyze_data(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        :param force_reload: Принудительная перезагрузка данных
        :return: результаты анализа данных
        """
        logger.info("Загружаем и анализируем исходные данные...")

        if self.raw_data is None or force_reload:
            self.raw_data = self.data_loader.load_data()

        analysis_results = self.data_loader.analyze_column_types()

        if self.config is None or force_reload:
            self.config = self.data_loader.get_default_config()

        logger.info("Анализ данных завершен")
        return analysis_results

    def preprocess_data(self, config: Optional[PreprocessingConfig] = None,
                       use_cache: bool = True) -> pd.DataFrame:
        """
        :param config: Конфигурация предобработки
        :param use_cache: Использовать ли кэширование
        :return: предобработанные данные
        """
        logger.info("Начинаем предобработку данных...")

        if config is None:
            if self.config is None:
                self.load_and_analyze_data()
            config = self.config
        else:
            self.config = config

        if use_cache and config.cache_enabled:
            if self.cache_manager.is_cache_valid(self.data_filepath, config):
                logger.info("Загружаем данные из кэша...")
                try:
                    processed_data, preprocessor, cached_config = self.cache_manager.load_from_cache()
                    self.processed_data = processed_data
                    self.preprocessor = preprocessor
                    self.config = cached_config

                    logger.info("Данные успешно загружены из кэша")
                    return processed_data
                except Exception as e:
                    logger.warning(f"Ошибка загрузки из кэша: {e}. Выполняем предобработку заново.")

        if self.raw_data is None:
            self.load_and_analyze_data()

        self.preprocessor = DataPreprocessor(config)
        self.processed_data = self.preprocessor.fit_transform(self.raw_data)

        if use_cache and config.cache_enabled:
            try:
                self.cache_manager.save_to_cache(
                    self.processed_data,
                    self.preprocessor,
                    self.data_filepath,
                    config
                )
                logger.info("Результаты сохранены в кэш")
            except Exception as e:
                logger.warning(f"Не удалось сохранить в кэш: {e}")

        self.processing_summary = self.preprocessor.get_preprocessing_summary()

        logger.info(f"Предобработка завершена. Получено {self.processed_data.shape[1]} признаков")

        return self.processed_data

    def select_informative_features(self, n_features: int,
                                  clustering_method: str = 'single_linkage',
                                  n_clusters: int = 3,
                                  distance_metric: str = 'euclidean') -> List[str]:
        """
        Выбор наиболее информативных признаков с использованием СПА.

        :param n_features: Количество признаков для выбора
        :param clustering_method: Метод кластеризации для оценки качества
        :param n_clusters: Количество кластеров
        :param distance_metric: Метрика расстояния
        :return: список выбранных признаков
        """
        if self.processed_data is None:
            raise ValueError("Сначала выполните предобработку данных")

        logger.info(f"Выбираем {n_features} наиболее информативных признаков...")

        selected_features, quality = self.feature_selector.select_features_spa(
            self.processed_data,
            n_features=n_features,
            clustering_method=clustering_method,
            n_clusters=n_clusters,
            distance_metric=distance_metric
        )

        self.selected_features = selected_features

        logger.info(f"Выбор признаков завершен. Качество: {quality:.6f}")

        return selected_features

    def get_selected_data(self) -> pd.DataFrame:
        """
        :return: данные с выбранными признаками
        """
        if self.selected_features is None:
            raise ValueError("Сначала выберите признаки с помощью select_informative_features()")

        if self.processed_data is None:
            raise ValueError("Предобработанные данные недоступны")

        return self.processed_data[self.selected_features]

    def analyze_k_anonymity(self, quasi_identifiers: Optional[List[str]] = None,
                          use_processed_data: bool = False) -> Dict[str, Any]:
        """
        :param quasi_identifiers: Список квази-идентификаторов
        :param use_processed_data: Использовать предобработанные данные
        :return: результаты анализа K-анонимности
        """
        logger.info("Начинаем анализ K-анонимности...")

        if use_processed_data:
            if self.processed_data is None:
                raise ValueError("Предобработанные данные недоступны. Выполните предобработку сначала.")
            analysis_data = self.processed_data
            logger.info("Используем предобработанные данные для анализа K-анонимности")
        else:
            if self.raw_data is None:
                self.load_and_analyze_data()
            analysis_data = self.raw_data
            logger.info("Используем исходные данные для анализа K-анонимности")

        if quasi_identifiers is None:
            if self.config is not None:
                quasi_identifiers = [
                    col_name for col_name, col_config in self.config.columns.items()
                    if col_config.is_quasi_identifier
                ]
            else:
                quasi_identifiers = ['age', 'gender', 'parental_education_level',
                                   'internet_quality', 'part_time_job']

        if not quasi_identifiers:
            logger.warning("Не определены квази-идентификаторы. Используем умолчания.")
            quasi_identifiers = ['age', 'gender']

        logger.info(f"Квази-идентификаторы для анализа: {quasi_identifiers}")

        k_anonymity_results = self.k_anonymity_analyzer.analyze_k_anonymity(
            analysis_data, quasi_identifiers
        )

        logger.info(f"Анализ K-анонимности завершен. K = {k_anonymity_results['k_anonymity_value']}")

        return k_anonymity_results

    def anonymize_data(self, quasi_identifiers=None, anonymization_level='moderate'):
        """
        Анонимизирует данные, применяя обобщение и маскирование к квази-идентификаторам.
        """
        if self.raw_data is None:
            self.load_and_analyze_data()

        if quasi_identifiers is None:
            # Используем дефолтные квази-идентификаторы из конфигурации
            if self.config is not None:
                quasi_identifiers = [
                    col_name for col_name, col_config in self.config.columns.items()
                    if col_config.is_quasi_identifier
                ]
            else:
                quasi_identifiers = ['age', 'gender', 'parental_education_level',
                                   'internet_quality', 'part_time_job']

        logger.info(f"Начинаем анонимизацию данных. Квази-идентификаторы: {quasi_identifiers}")

        # Копируем данные, чтобы не изменять оригинал
        anonymized_data = self.raw_data.copy()

        # Применяем обобщение и/или маскирование для каждого квази-идентификатора
        for column in quasi_identifiers:
            if column not in anonymized_data.columns:
                logger.warning(f"Столбец {column} отсутствует в данных, пропускаем")
                continue

            # Обработка в зависимости от типа данных
            if pd.api.types.is_numeric_dtype(anonymized_data[column]):
                anonymized_data = self._generalize_numeric_column(
                    anonymized_data, column, anonymization_level
                )
            elif column.lower() in ['gender', 'sex']:
                # Для пола делаем более простые категории, если уровень высокий
                if anonymization_level == 'high':
                    anonymized_data[column] = '***'
                elif anonymization_level == 'moderate' and 'gender' in column.lower():
                    # Заменяем на более общие категории
                    gender_map = {'Male': 'M', 'Female': 'F', 'Other': 'O'}
                    anonymized_data[column] = anonymized_data[column].astype(str).map(
                        lambda x: gender_map.get(str(x), str(x))
                    )
            elif 'education' in column.lower():
                # Обобщаем уровень образования
                if anonymization_level == 'high':
                    # Только высшее/не высшее
                    anonymized_data[column] = anonymized_data[column].astype(str).map(
                        lambda x: 'Higher Education' if 'bachelor' in str(x).lower()
                                                      or 'master' in str(x).lower()
                                                      or 'phd' in str(x).lower()
                                                     else 'Other'
                    )
                elif anonymization_level == 'moderate':
                    # Более детальные категории, но всё равно обобщенные
                    edu_map = {
                        'None': 'No Formal Education',
                        'Primary': 'Basic Education',
                        'Secondary': 'Basic Education',
                        'High School': 'Secondary Education',
                        'Bachelor': 'Higher Education',
                        'Master': 'Higher Education',
                        'PhD': 'Higher Education'
                    }
                    anonymized_data[column] = anonymized_data[column].astype(str).map(
                        lambda x: edu_map.get(str(x), str(x))
                    )
            elif 'quality' in column.lower() or 'rating' in column.lower():
                # Обобщаем рейтинги и качество
                if anonymization_level in ['moderate', 'high']:
                    # Сокращаем количество категорий
                    anonymized_data[column] = anonymized_data[column].astype(str).map(
                        lambda x: 'High' if str(x) in ['5', '4', 'Excellent', 'Good']
                                         else ('Medium' if str(x) in ['3', 'Average']
                                               else 'Low')
                    )
            else:
                # Для других категориальных столбцов
                if anonymization_level == 'high':
                    anonymized_data[column] = '***'

        # После анонимизации отдельных столбцов проверяем k-анонимность
        if len(quasi_identifiers) > 0:
            k_anonymity_results = self._calculate_k_anonymity(anonymized_data, quasi_identifiers)
            logger.info(f"K-анонимность после обобщения: {k_anonymity_results['k_anonymity_value']}")

            # Только если k-анонимность очень низкая и уровень анонимизации высокий,
            # удаляем некоторые строки для повышения k-анонимности
            if k_anonymity_results['k_anonymity_value'] < 2 and anonymization_level == 'high':
                logger.info(f"K-анонимность слишком низкая, удаляем редкие комбинации")
                anonymized_data = self._remove_rare_combinations(
                    anonymized_data, quasi_identifiers, min_count=2
                )
            # Для moderate уровня делаем более мягкое удаление
            elif k_anonymity_results['k_anonymity_value'] < 1 and anonymization_level == 'moderate':
                logger.info(f"K-анонимность низкая, удаляем только уникальные комбинации")
                anonymized_data = self._remove_rare_combinations(
                    anonymized_data, quasi_identifiers, min_count=1
                )

        logger.info("Анонимизация данных завершена")
        return anonymized_data

    def _generalize_numeric_column(self, df, column, anonymization_level):
        """
        Обобщает числовой столбец, разбивая его на категории.
        Исправленная версия для работы с строковыми значениями вместо категориальных.
        """
        df = df.copy()

        # Определяем количество категорий в зависимости от уровня анонимизации
        if anonymization_level == 'low':
            num_categories = 8
        elif anonymization_level == 'moderate':
            num_categories = 4
        else:  # high
            num_categories = 2

        # Преобразуем столбец в числовой формат
        try:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        except Exception as e:
            logger.warning(f"Не удалось преобразовать столбец {column} в числовой формат: {e}")
            return df

        # Отбрасываем NaN значения для вычисления квантилей
        valid_values = df[column].dropna()

        # Проверяем, достаточно ли у нас данных
        if len(valid_values) < num_categories + 1:
            logger.warning(f"Недостаточно уникальных значений в столбце {column} для разбиения на {num_categories} категорий")
            return df

        try:
            # Вычисляем квантильные точки для разбиения
            quantiles = [i / num_categories for i in range(num_categories + 1)]
            boundaries = valid_values.quantile(quantiles).values

            # Применяем функцию для округления границ
            boundaries = self._beautify_boundaries(boundaries)

            # Обеспечиваем уникальность и возрастающий порядок границ
            boundaries = self._ensure_increasing(sorted(set(boundaries)))

            # Проверяем, что границы покрывают весь диапазон данных
            min_value = valid_values.min()
            max_value = valid_values.max()
            if boundaries[0] > min_value:
                boundaries[0] = min_value
            if boundaries[-1] < max_value:
                boundaries[-1] = max_value

            # Создаем метки для категорий
            labels = []
            for i in range(len(boundaries) - 1):
                lower = boundaries[i]
                upper = boundaries[i + 1]
                labels.append(f"{lower}-{upper}")

            # Разбиваем данные на категории и сразу преобразуем в строки
            # Важно: pd.cut возвращает Categorical, поэтому преобразуем в строки
            result = pd.cut(df[column], bins=boundaries, labels=labels, include_lowest=True)
            df[column] = result.astype(str)

        except Exception as e:
            logger.warning(f"Ошибка при обобщении столбца {column}: {e}")
            # Если возникла ошибка, преобразуем столбец в строки
            df[column] = df[column].astype(str)

        return df

    def _beautify_boundaries(self, boundaries):
        """
        Округляет границы категорий до "красивых" чисел.
        """
        beautified = []
        for b in boundaries:
            if b == 0:
                beautified.append(0)
                continue
            elif b < 1:
                # Округляем до 1 знака после запятой
                b_rounded = round(b, 1)
            elif b < 10:
                # Округляем до ближайшего целого числа
                b_rounded = round(b)
            elif b < 100:
                # Округляем до ближайшего кратного 5
                b_rounded = round(b / 5) * 5
            elif b < 1000:
                # Округляем до ближайшего кратного 50
                b_rounded = round(b / 50) * 50
            else:
                # Округляем до ближайшего кратного 500
                b_rounded = round(b / 500) * 500
            beautified.append(b_rounded)
        return beautified

    def _ensure_increasing(self, boundaries):
        """
        Обеспечивает, что границы строго возрастают.
        """
        for i in range(1, len(boundaries)):
            if boundaries[i] <= boundaries[i - 1]:
                boundaries[i] = boundaries[i - 1] + 1e-6
        return boundaries

    def _calculate_k_anonymity(self, df, quasi_identifiers):
        """
        Рассчитывает k-анонимность для указанных квази-идентификаторов.
        Исправленная версия с преобразованием категориальных столбцов в строки.
        """
        # Создаем копию для безопасности
        df_copy = df.copy()

        # Преобразуем все столбцы в строки для корректной группировки
        for col in quasi_identifiers:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].astype(str)

        # Группируем по квази-идентификаторам
        try:
            grouped = df_copy.groupby(quasi_identifiers, observed=False).size().reset_index(name='group_size')
        except Exception as e:
            logger.warning(f"Ошибка при группировке для расчета k-анонимности: {e}")
            return {
                'k_anonymity_value': 0,
                'total_records': len(df),
                'equivalence_classes_count': 0,
                'group_size_stats': {
                    'min': 0,
                    'max': 0,
                    'mean': 0,
                    'median': 0
                }
            }

        # Вычисляем k-анонимность
        k_anonymity_value = grouped['group_size'].min() if not grouped.empty else 0

        return {
            'k_anonymity_value': int(k_anonymity_value),
            'total_records': len(df),
            'equivalence_classes_count': len(grouped),
            'group_size_stats': {
                'min': int(grouped['group_size'].min()) if not grouped.empty else 0,
                'max': int(grouped['group_size'].max()) if not grouped.empty else 0,
                'mean': float(grouped['group_size'].mean()) if not grouped.empty else 0,
                'median': float(grouped['group_size'].median()) if not grouped.empty else 0
            }
        }

    def _remove_rare_combinations(self, df, quasi_identifiers, min_count=3):
        """
        Удаляет строки с редкими комбинациями квази-идентификаторов.
        Исправленная версия с преобразованием категориальных столбцов в строки.
        """
        # Копируем DataFrame
        df_copy = df.copy()

        # Проверяем, есть ли в таблице хотя бы min_count * 3 строк
        if len(df_copy) < min_count * 3:
            logger.warning(f"Набор данных слишком мал ({len(df_copy)} строк) для удаления редких комбинаций")
            return df_copy

        # Преобразуем все столбцы в строки для корректной группировки
        for col in quasi_identifiers:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].astype(str)

        # Группируем по квази-идентификаторам
        try:
            group_counts = df_copy.groupby(quasi_identifiers, observed=False).size()
        except Exception as e:
            logger.warning(f"Ошибка при группировке по квази-идентификаторам: {e}")
            return df_copy

        # Фильтруем группы, где количество записей меньше min_count
        rare_groups = group_counts[group_counts < min_count]

        # Проверяем, есть ли строки для удаления
        if len(rare_groups) == 0:
            logger.info("Нет редких комбинаций для удаления")
            return df_copy

        rows_to_remove_count = rare_groups.sum()
        removal_percentage = (rows_to_remove_count / len(df)) * 100

        # Если процент удаляемых строк слишком высок, уменьшаем требования
        if removal_percentage > 25:
            logger.warning(f"Слишком много строк ({removal_percentage:.2f}%) будет удалено. Снижаем требования.")
            min_count = max(1, min_count - 1)
            rare_groups = group_counts[group_counts < min_count]
            rows_to_remove_count = rare_groups.sum()
            removal_percentage = (rows_to_remove_count / len(df)) * 100

            # Если всё ещё слишком много, возвращаем исходные данные
            if removal_percentage > 25:
                logger.warning(f"Даже после снижения требований, слишком много строк будет удалено. Оставляем как есть.")
                return df_copy

        # Индексы строк для удаления
        rows_to_remove = []

        # Для каждой редкой группы находим соответствующие строки
        for group_values in rare_groups.index:
            group_values = [group_values] if not isinstance(group_values, tuple) else group_values

            # Проверка длины группы и квази-идентификаторов
            if len(group_values) != len(quasi_identifiers):
                logger.warning(f"Несоответствие размеров: {len(group_values)} vs {len(quasi_identifiers)}")
                continue

            # Создаем маску для поиска строк с данной комбинацией квази-идентификаторов
            mask = pd.Series(True, index=df_copy.index)
            for i, col in enumerate(quasi_identifiers):
                if col in df_copy.columns:
                    mask = mask & (df_copy[col] == group_values[i])

            # Добавляем найденные индексы
            rows_to_remove.extend(df_copy[mask].index.tolist())

        # Удаляем идентифицированные строки
        df_filtered = df_copy.drop(rows_to_remove)

        logger.info(f"Удалено {len(rows_to_remove)} строк с редкими комбинациями "
                   f"({len(rows_to_remove) / len(df) * 100:.2f}% от общего числа)")

        return df_filtered

    def anonymize_and_cluster(self, n_features=None, k_clusters=3, metric='euclidean',
                             anonymization_level='moderate'):
        """
        Анонимизирует данные, затем выбирает признаки и выполняет кластеризацию.
        Исправленная версия с более надежной обработкой ошибок.
        """
        if self.raw_data is None:
            self.load_and_analyze_data()

        logger.info("Запуск анонимизации с последующей кластеризацией")

        # Анонимизируем данные
        anonymized_data = self.anonymize_data(anonymization_level=anonymization_level)

        # Специальная предобработка для анонимизированных данных
        processed_anon_data = self._preprocess_anonymized_data(anonymized_data)

        # Проверка на пустые данные
        if processed_anon_data.empty:
            logger.error("После предобработки анонимизированных данных получен пустой DataFrame")
            raise ValueError("После предобработки анонимизированных данных получен пустой DataFrame")

        # Выбираем признаки, если указано
        if n_features is not None and n_features < processed_anon_data.shape[1]:
            try:
                selected_features = self.feature_selector.select_features_spa(
                    processed_anon_data,
                    n_features=n_features,
                    clustering_method='single_linkage',
                    n_clusters=k_clusters,
                    distance_metric=metric
                )[0]
                data_for_clustering = processed_anon_data[selected_features]
            except Exception as e:
                logger.warning(f"Ошибка при выборе признаков: {e}")
                # Используем все признаки в случае ошибки
                data_for_clustering = processed_anon_data
        else:
            data_for_clustering = processed_anon_data

        # Выполняем кластеризацию
        try:
            from mathematical_algorithms import single_linkage_clustering
            labels, _ = single_linkage_clustering(
                data_for_clustering.values,
                n_clusters=k_clusters,
                metric=metric,
                verbose=False
            )
            logger.info(f"Кластеризация анонимизированных данных успешно выполнена: {k_clusters} кластеров")
            return labels, data_for_clustering
        except Exception as e:
            logger.error(f"Ошибка при кластеризации анонимизированных данных: {e}")
            raise

    def compare_clustering_results(self, k_clusters=3, n_features=5, metric='euclidean',
                                  anonymization_level='moderate'):
        """
        Сравнивает результаты различных подходов к кластеризации,
        вычисляя индекс Фоулкса-Мэллова между ними.

        :param k_clusters: Количество кластеров
        :param n_features: Количество признаков для выбора
        :param metric: Метрика расстояния
        :param anonymization_level: Уровень анонимизации
        :return: Словарь с результатами сравнения
        """
        try:
            from mathematical_algorithms import fowlkes_mallows_index, single_linkage_clustering
        except ImportError:
            logger.error("Не удалось импортировать необходимые функции")
            return {"error": "Не удалось импортировать необходимые функции"}

        results = {}

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
        labels_anonymized, _ = self.anonymize_and_cluster(
            n_features=n_features,
            k_clusters=k_clusters,
            metric=metric,
            anonymization_level=anonymization_level
        )
        results["labels_anonymized"] = labels_anonymized

        # Вычисляем индексы Фоулкса-Мэллова для всех пар результатов
        fm_preproc_vs_features = fowlkes_mallows_index(labels_preprocessing, labels_features)
        fm_preproc_vs_anon = fowlkes_mallows_index(labels_preprocessing, labels_anonymized)
        fm_features_vs_anon = fowlkes_mallows_index(labels_features, labels_anonymized)

        results["fm_indices"] = {
            "preprocessing_vs_features": fm_preproc_vs_features,
            "preprocessing_vs_anonymized": fm_preproc_vs_anon,
            "features_vs_anonymized": fm_features_vs_anon
        }

        logger.info(f"Результаты сравнения кластеризации: {results['fm_indices']}")

        return results

    def _preprocess_anonymized_data(self, anonymized_data):
        """
        Специальная предобработка для анонимизированных данных.
        Преобразует категориальные и диапазонные данные в числовой формат.
        """
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder

        processed_data = pd.DataFrame(index=anonymized_data.index)

        # Обрабатываем каждый столбец
        for column in anonymized_data.columns:
            # Получаем данные колонки и преобразуем в строки, если категориальные
            column_data = anonymized_data[column]
            if pd.api.types.is_categorical_dtype(column_data):
                column_data = column_data.astype(str)

            # Проверяем, является ли столбец числовым или обобщенным диапазоном
            if pd.api.types.is_numeric_dtype(column_data):
                # Если числовой, добавляем как есть
                processed_data[column] = column_data
            elif isinstance(column_data.iloc[0], str) and column_data.str.contains('-').any():
                # Это диапазон, извлекаем среднее значение
                try:
                    def extract_midpoint(range_str):
                        if pd.isna(range_str) or not isinstance(range_str, str):
                            return np.nan
                        if '-' not in range_str:
                            return np.nan

                        parts = range_str.split('-')
                        if len(parts) != 2:
                            return np.nan

                        try:
                            lower = float(parts[0].strip())
                            upper = float(parts[1].strip())
                            return (lower + upper) / 2
                        except (ValueError, TypeError):
                            return np.nan

                    # Применяем функцию к столбцу
                    processed_data[column] = column_data.apply(extract_midpoint)

                    # Если все результаты NaN, используем порядковое кодирование
                    if processed_data[column].isna().all():
                        encoder = LabelEncoder()
                        processed_data[column] = encoder.fit_transform(column_data.fillna('unknown'))
                except Exception as e:
                    logger.warning(f"Ошибка обработки диапазона в столбце {column}: {e}")
                    # Запасной вариант - кодирование меток
                    try:
                        encoder = LabelEncoder()
                        processed_data[column] = encoder.fit_transform(column_data.fillna('unknown'))
                    except Exception as inner_e:
                        logger.warning(f"Вторичная ошибка кодирования столбца {column}: {inner_e}")
                        # Если и это не сработало, пропускаем столбец
                        continue
            else:
                # Категориальный столбец - используем порядковое кодирование
                try:
                    encoder = LabelEncoder()
                    processed_data[column] = encoder.fit_transform(column_data.fillna('unknown'))
                except Exception as e:
                    logger.warning(f"Ошибка кодирования столбца {column}: {e}")
                    # Пропускаем столбец
                    continue

        # Проверяем наличие NaN и заменяем на 0 (для нечисловых столбцов мы уже обработали NaN)
        for column in processed_data.columns:
            if processed_data[column].isna().any():
                processed_data[column] = processed_data[column].fillna(0)

        logger.info(f"Предобработка анонимизированных данных завершена: {processed_data.shape}")
        return processed_data
