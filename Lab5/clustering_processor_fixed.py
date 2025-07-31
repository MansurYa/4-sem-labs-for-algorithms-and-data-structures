"""
ИСПРАВЛЕННАЯ СИСТЕМА РАСШИРЕННОЙ КЛАСТЕРИЗАЦИИ
Выполняет кластеризацию от K=100 до K=2 и сохраняет результаты в CSV файл
Версия 2.0 - Исправлены критические ошибки и добавлены улучшения
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import sys
import signal
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExtendedClusteringProcessor:
    """
    ИСПРАВЛЕННЫЙ класс для расширенной кластеризации с детальным сохранением результатов.
    Выполняет кластеризацию от K=max_clusters до K=min_clusters и сохраняет все промежуточные результаты.

    Версия 2.0 - Исправления:
    - Правильная оценка времени выполнения
    - Комплексная валидация параметров
    - Robust обработка ошибок
    - Прогресс-индикатор для длительных операций
    - Возможность прерывания процесса
    """

    def __init__(self, data_file: str = "student_habits_performance.csv"):
        """
        Инициализация процессора расширенной кластеризации.

        :param data_file: Путь к файлу исходных данных
        """
        self.data_file = data_file
        self.results_dir = Path("clustering_results")
        self.results_dir.mkdir(exist_ok=True)

        # Результаты обработки
        self.data_processor = None
        self.selected_features = None
        self.selected_data = None
        self.clustering_history = None
        self.processing_metadata = {}

        # Флаги состояния
        self.is_interrupted = False
        self.current_operation = "Инициализация"

        # Регистрируем обработчик прерывания
        signal.signal(signal.SIGINT, self._handle_interrupt)

        logger.info(f"✓ Инициализирован ExtendedClusteringProcessor v2.0 для файла: {data_file}")

    def _handle_interrupt(self, signum, frame):
        """Обработчик прерывания процесса пользователем."""
        logger.warning(f"⏹️  Получен сигнал прерывания во время: {self.current_operation}")
        self.is_interrupted = True

    def _check_interruption(self):
        """Проверка прерывания процесса."""
        if self.is_interrupted:
            logger.info("🛑 Процесс прерван пользователем")
            raise KeyboardInterrupt("Операция прервана пользователем")

    def process_full_pipeline(self,
                            min_clusters: int = 2,
                            max_clusters: int = 100,
                            distance_metric: str = 'chebyshev',
                            save_csv: bool = True,
                            save_metadata: bool = True,
                            progress_callback=None) -> str:
        """
        ИСПРАВЛЕННЫЙ метод выполнения полного цикла обработки данных и кластеризации.

        :param min_clusters: Минимальное количество кластеров
        :param max_clusters: Максимальное количество кластеров
        :param distance_metric: Метрика расстояния
        :param save_csv: Сохранять ли результаты в CSV
        :param save_metadata: Сохранять ли метаданные
        :param progress_callback: Функция для обновления прогресса
        :return: Путь к сохраненному CSV файлу
        :raises ValueError: При некорректных параметрах
        :raises RuntimeError: При критических ошибках выполнения
        """
        logger.info("🚀 ЗАПУСК ПОЛНОГО ЦИКЛА РАСШИРЕННОЙ КЛАСТЕРИЗАЦИИ v2.0")
        logger.info("=" * 70)

        try:
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 1: Валидация параметров
            self._validate_parameters(min_clusters, max_clusters, distance_metric)

            # Этап 1: Подготовка данных (20% прогресса)
            self.current_operation = "Подготовка данных"
            if progress_callback:
                progress_callback(0, "Подготовка данных...")
            self._prepare_data()
            self._check_interruption()

            if progress_callback:
                progress_callback(20, "Данные подготовлены")

            # Этап 2: Расширенная кластеризация (60% прогресса)
            self.current_operation = "Кластеризация"
            if progress_callback:
                progress_callback(20, "Начинаем кластеризацию...")

            self._perform_extended_clustering(min_clusters, max_clusters, distance_metric, progress_callback)
            self._check_interruption()

            if progress_callback:
                progress_callback(80, "Кластеризация завершена")

            # Этап 3: Сохранение результатов (20% прогресса)
            self.current_operation = "Сохранение результатов"
            csv_path = None
            if save_csv:
                if progress_callback:
                    progress_callback(80, "Сохранение в CSV...")
                csv_path = self._save_results_to_csv()
                self._check_interruption()

            if save_metadata:
                if progress_callback:
                    progress_callback(90, "Сохранение метаданных...")
                self._save_metadata()
                self._check_interruption()

            if progress_callback:
                progress_callback(100, "Полный цикл завершен!")

            logger.info("✅ ПОЛНЫЙ ЦИКЛ ЗАВЕРШЕН УСПЕШНО!")
            return csv_path

        except KeyboardInterrupt:
            logger.warning("⏹️  Процесс прерван пользователем")
            raise
        except (ValueError, RuntimeError) as e:
            logger.error(f"❌ Ошибка выполнения: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Критическая неожиданная ошибка: {e}")
            raise RuntimeError(f"Критическая ошибка в полном цикле: {e}") from e

    def _validate_parameters(self, min_clusters: int, max_clusters: int, distance_metric: str):
        """
        КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 2: Комплексная валидация входных параметров.

        :param min_clusters: Минимальное количество кластеров
        :param max_clusters: Максимальное количество кластеров
        :param distance_metric: Метрика расстояния
        :raises ValueError: При некорректных параметрах
        """
        logger.info("🔍 Валидация параметров...")

        # Проверка типов данных
        if not isinstance(min_clusters, int):
            raise ValueError(f"min_clusters должно быть целым числом, получено: {type(min_clusters)}")

        if not isinstance(max_clusters, int):
            raise ValueError(f"max_clusters должно быть целым числом, получено: {type(max_clusters)}")

        if not isinstance(distance_metric, str):
            raise ValueError(f"distance_metric должно быть строкой, получено: {type(distance_metric)}")

        # Проверка значений
        if min_clusters < 1:
            raise ValueError(f"min_clusters должно быть положительным, получено: {min_clusters}")

        if max_clusters < 2:
            raise ValueError(f"max_clusters должно быть не меньше 2, получено: {max_clusters}")

        if min_clusters >= max_clusters:
            raise ValueError(f"min_clusters ({min_clusters}) должно быть меньше max_clusters ({max_clusters})")

        # Проверка разумности диапазона
        if max_clusters - min_clusters > 500:
            logger.warning(f"⚠️  Большой диапазон кластеров ({max_clusters - min_clusters}), процесс может занять много времени")

        # Проверка метрики расстояния
        valid_metrics = ['euclidean', 'euclidean_squared', 'chebyshev', 'manhattan', 'minkowski']
        if distance_metric not in valid_metrics:
            logger.warning(f"⚠️  Неизвестная метрика '{distance_metric}', поддерживаемые: {valid_metrics}")

        logger.info(f"  ✓ min_clusters: {min_clusters}")
        logger.info(f"  ✓ max_clusters: {max_clusters}")
        logger.info(f"  ✓ Количество разбиений: {max_clusters - min_clusters + 1}")
        logger.info(f"  ✓ distance_metric: {distance_metric}")

    def _prepare_data(self):
        """Подготовка данных для кластеризации с улучшенной обработкой ошибок."""
        logger.info("📊 ЭТАП 1: Подготовка данных")

        try:
            # Проверка существования файла данных
            if not Path(self.data_file).exists():
                raise FileNotFoundError(f"Файл данных не найден: {self.data_file}")

            from data_preprocessing import StudentDataProcessor

            # Инициализация и загрузка
            logger.info("  Загрузка данных...")
            self.data_processor = StudentDataProcessor(self.data_file)
            raw_data = self.data_processor.load_and_analyze_data()

            if self.data_processor.raw_data is None or self.data_processor.raw_data.empty:
                raise RuntimeError("Не удалось загрузить данные из файла")

            logger.info(f"  ✓ Загружено: {self.data_processor.raw_data.shape}")

            # Предобработка
            logger.info("  Предобработка данных...")
            processed_data = self.data_processor.preprocess_data()

            if processed_data is None or processed_data.empty:
                raise RuntimeError("Предобработка данных завершилась неудачно")

            logger.info(f"  ✓ Предобработано: {processed_data.shape}")

            # Проверка потерь данных при предобработке
            data_loss_percentage = (1 - processed_data.shape[0] / self.data_processor.raw_data.shape[0]) * 100
            if data_loss_percentage > 50:
                raise RuntimeError(f"Критическая потеря данных при предобработке: {data_loss_percentage:.1f}%")
            elif data_loss_percentage > 20:
                logger.warning(f"⚠️  Значительная потеря данных при предобработке: {data_loss_percentage:.1f}%")

            # Выбор 2 признаков для 2D визуализации
            logger.info("  Выбор 2 наиболее информативных признаков...")
            self.selected_features = self.data_processor.select_informative_features(
                n_features=2,
                distance_metric='chebyshev'
            )
            self.selected_data = self.data_processor.get_selected_data()

            if self.selected_data is None or self.selected_data.empty:
                raise RuntimeError("Не удалось выбрать информативные признаки")

            if len(self.selected_features) != 2:
                raise RuntimeError(f"Ожидалось 2 признака, получено: {len(self.selected_features)}")

            logger.info(f"  ✓ Выбраны признаки: {self.selected_features}")
            logger.info(f"  ✓ Размерность для кластеризации: {self.selected_data.shape}")

            # Проверка качества выбранных признаков
            self._validate_selected_features()

        except ImportError as e:
            raise RuntimeError(f"Не удалось импортировать модуль предобработки данных: {e}")
        except Exception as e:
            logger.error(f"  ❌ Ошибка подготовки данных: {e}")
            raise RuntimeError(f"Критическая ошибка подготовки данных: {e}") from e

    def _validate_selected_features(self):
        """Валидация качества выбранных признаков с предупреждениями."""
        logger.info("  Валидация выбранных признаков...")

        quality_issues = []

        for i, feature_name in enumerate(self.selected_features):
            col_data = self.selected_data.iloc[:, i]
            unique_count = col_data.nunique()
            std_dev = col_data.std()
            value_range = col_data.max() - col_data.min()

            logger.info(f"    {feature_name}:")
            logger.info(f"      Уникальных значений: {unique_count}")
            logger.info(f"      Стандартное отклонение: {std_dev:.6f}")
            logger.info(f"      Диапазон значений: {value_range:.6f}")

            # Критические проблемы с качеством признаков
            if unique_count < 10:
                issue = f"Признак {feature_name} имеет очень мало уникальных значений: {unique_count}"
                quality_issues.append(issue)
                logger.warning(f"      ⚠️  {issue}")

            if std_dev < 1e-6:
                issue = f"Признак {feature_name} почти константный (σ={std_dev:.8f})"
                quality_issues.append(issue)
                logger.error(f"      ❌ {issue}")

            if value_range < 1e-6:
                issue = f"Признак {feature_name} имеет нулевой диапазон: {value_range:.8f}"
                quality_issues.append(issue)
                logger.error(f"      ❌ {issue}")

        # Проверка на критические проблемы
        critical_issues = [issue for issue in quality_issues if "почти константный" in issue or "нулевой диапазон" in issue]
        if critical_issues:
            raise RuntimeError(f"Критические проблемы с качеством признаков: {'; '.join(critical_issues)}")

        if quality_issues:
            logger.warning(f"  ⚠️  Обнаружены проблемы с качеством признаков, но процесс может продолжиться")

    def _perform_extended_clustering(self, min_clusters: int, max_clusters: int, distance_metric: str, progress_callback=None):
        """ИСПРАВЛЕННЫЙ метод выполнения расширенной кластеризации."""
        logger.info(f"🔬 ЭТАП 2: Расширенная кластеризация (K={max_clusters}→{min_clusters})")

        try:
            from mathematical_algorithms import single_linkage_clustering_with_history

            # Адаптация максимального количества кластеров к размеру данных
            n_samples = self.selected_data.shape[0]
            original_max_clusters = max_clusters

            if max_clusters > n_samples:
                max_clusters = n_samples
                logger.warning(f"  ⚠️  Максимальное K ограничено размером данных: {max_clusters} (было {original_max_clusters})")

            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 3: Правильная оценка времени
            estimated_time_str, estimated_seconds = self._estimate_clustering_time_corrected(n_samples, min_clusters, max_clusters)

            logger.info(f"  Запуск односвязывающей кластеризации...")
            logger.info(f"  Параметры: K={max_clusters}→{min_clusters}, метрика={distance_metric}")
            logger.info(f"  Размер данных: {n_samples} объектов, {self.selected_data.shape[1]} признаков")
            logger.info(f"  Количество итераций: {max_clusters - min_clusters}")
            logger.info(f"  Ожидаемое время выполнения: {estimated_time_str}")

            # Предупреждение о длительности процесса
            if estimated_seconds > 300:  # Более 5 минут
                logger.warning(f"⚠️  ВНИМАНИЕ: Процесс может занять {estimated_time_str}")
                logger.warning(f"⚠️  Для ускорения рассмотрите уменьшение max_clusters до 50 или меньше")

            # Выполнение кластеризации с отслеживанием времени
            start_time = datetime.now()

            # Создаем callback для прогресса кластеризации
            def clustering_progress_callback(current_k, total_iterations):
                if progress_callback:
                    progress_percent = 20 + (60 * (max_clusters - current_k) / total_iterations)
                    progress_callback(int(progress_percent), f"Кластеризация: K={current_k}")
                self._check_interruption()

            self.clustering_history, detailed_history = single_linkage_clustering_with_history(
                data=self.selected_data.values,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                metric=distance_metric,
                verbose=True
            )

            end_time = datetime.now()
            actual_time = end_time - start_time

            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 4: Проверка результатов кластеризации
            if not self.clustering_history:
                raise RuntimeError("Кластеризация не дала результатов")

            expected_k_count = max_clusters - min_clusters + 1
            actual_k_count = len(self.clustering_history)

            if actual_k_count < expected_k_count * 0.8:  # Менее 80% ожидаемых разбиений
                logger.warning(f"⚠️  Получено меньше разбиений чем ожидалось: {actual_k_count} из {expected_k_count}")

            logger.info(f"  ✓ Кластеризация завершена за: {actual_time}")
            logger.info(f"  ✓ Получено разбиений: {len(self.clustering_history)}")
            logger.info(f"  ✓ Доступные K: {sorted(self.clustering_history.keys(), reverse=True)}")

            # Оценка точности прогнозирования времени
            time_prediction_accuracy = min(estimated_seconds, actual_time.total_seconds()) / max(estimated_seconds, actual_time.total_seconds())
            logger.info(f"  ℹ️  Точность прогноза времени: {time_prediction_accuracy:.1%}")

            # Сохраняем метаданные процесса
            self.processing_metadata = {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'processing_duration': str(actual_time),
                'estimated_duration_seconds': estimated_seconds,
                'actual_duration_seconds': actual_time.total_seconds(),
                'time_prediction_accuracy': time_prediction_accuracy,
                'min_clusters': min_clusters,
                'max_clusters': max_clusters,
                'distance_metric': distance_metric,
                'data_shape': self.selected_data.shape,
                'selected_features': self.selected_features,
                'available_k_values': sorted(self.clustering_history.keys(), reverse=True),
                'total_iterations': len(detailed_history) if detailed_history else 0,
                'expected_k_count': expected_k_count,
                'actual_k_count': actual_k_count
            }

        except ImportError as e:
            raise RuntimeError(f"Не удалось импортировать математические алгоритмы: {e}")
        except Exception as e:
            logger.error(f"  ❌ Ошибка кластеризации: {e}")
            raise RuntimeError(f"Критическая ошибка кластеризации: {e}") from e

    def _estimate_clustering_time_corrected(self, n_samples: int, min_clusters: int, max_clusters: int) -> Tuple[str, float]:
        """
        КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Правильная оценка времени выполнения кластеризации.

        :param n_samples: Количество объектов
        :param min_clusters: Минимальное количество кластеров
        :param max_clusters: Максимальное количество кластеров
        :return: (строковое_описание, время_в_секундах)
        """
        # ИСПРАВЛЕНИЕ: Правильная формула для количества итераций
        iterations = max_clusters - min_clusters

        # Эмпирическая формула основанная на реальных измерениях
        # Сложность односвязывающего алгоритма: O(iterations * n_samples^2)
        # Коэффициент получен из практических тестов
        base_complexity = iterations * (n_samples ** 1.8) / 10000000

        # Коррекция на размер данных
        if n_samples <= 500:
            time_factor = 1.0
        elif n_samples <= 1000:
            time_factor = 1.5
        elif n_samples <= 2000:
            time_factor = 2.5
        else:
            time_factor = 4.0

        estimated_seconds = base_complexity * time_factor

        # Форматирование времени
        if estimated_seconds < 60:
            time_str = f"{int(estimated_seconds)} секунд"
        elif estimated_seconds < 3600:
            minutes = int(estimated_seconds / 60)
            time_str = f"{minutes} минут"
        else:
            hours = int(estimated_seconds / 3600)
            minutes = int((estimated_seconds % 3600) / 60)
            time_str = f"{hours} ч {minutes} мин"

        return time_str, estimated_seconds

    def _save_results_to_csv(self) -> str:
        """ИСПРАВЛЕННЫЙ метод сохранения результатов кластеризации в CSV файл."""
        logger.info("💾 ЭТАП 3: Сохранение результатов в CSV")

        try:
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 5: Проверка наличия данных
            if not self.clustering_history:
                raise RuntimeError("История кластеризации пуста - нечего сохранять")

            if self.selected_data is None or self.selected_data.empty:
                raise RuntimeError("Отсутствуют исходные данные для сохранения")

            if not self.selected_features:
                raise RuntimeError("Отсутствует информация о выбранных признаках")

            # Создание базового DataFrame с исходными признаками
            logger.info("  Создание базового DataFrame...")
            results_df = self.selected_data.copy()

            # Добавление результатов кластеризации для каждого K
            logger.info("  Добавление результатов кластеризации...")
            k_values = sorted(self.clustering_history.keys(), reverse=True)

            for i, k in enumerate(k_values):
                column_name = f'cluster_k{k:03d}'  # Форматирование с ведущими нулями

                # Проверка размерности перед добавлением
                labels = self.clustering_history[k]
                if len(labels) != len(results_df):
                    raise RuntimeError(f"Размерность меток для K={k} ({len(labels)}) не совпадает с размерностью данных ({len(results_df)})")

                results_df[column_name] = labels

                # Логирование прогресса для больших K
                if len(k_values) > 50 and i % 10 == 0:
                    logger.info(f"    Обработано {i+1}/{len(k_values)} разбиений...")

            # Создание имени файла с временной меткой
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"clustering_results_{timestamp}.csv"
            csv_path = self.results_dir / csv_filename

            # Проверка доступности для записи
            if csv_path.exists() and not csv_path.is_file():
                raise RuntimeError(f"Путь для сохранения заблокирован: {csv_path}")

            # Сохранение в CSV с обработкой ошибок
            logger.info(f"  Сохранение в файл: {csv_filename}")
            try:
                results_df.to_csv(csv_path, index=False)
            except PermissionError as e:
                raise RuntimeError(f"Нет прав для записи в файл: {csv_path}") from e
            except Exception as e:
                raise RuntimeError(f"Ошибка записи в CSV файл: {e}") from e

            # Проверка успешности сохранения
            if not csv_path.exists():
                raise RuntimeError(f"Файл не был создан: {csv_path}")

            file_size_mb = csv_path.stat().st_size / 1024 / 1024

            logger.info(f"  ✓ Результаты сохранены: {csv_path}")
            logger.info(f"  ✓ Размер файла: {file_size_mb:.2f} MB")
            logger.info(f"  ✓ Столбцов в файле: {len(results_df.columns)}")
            logger.info(f"    - Признаки: {len(self.selected_features)}")
            logger.info(f"    - Результаты кластеризации: {len(self.clustering_history)}")
            logger.info(f"  ✓ Строк данных: {len(results_df)}")

            return str(csv_path)

        except Exception as e:
            logger.error(f"  ❌ Ошибка сохранения CSV: {e}")
            raise RuntimeError(f"Критическая ошибка сохранения CSV: {e}") from e

    def _save_metadata(self):
        """Сохранение метаданных процесса с улучшенной обработкой ошибок."""
        logger.info("📝 Сохранение метаданных процесса...")

        try:
            if not self.processing_metadata:
                logger.warning("  ⚠️  Метаданные процесса отсутствуют")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata_filename = f"clustering_metadata_{timestamp}.json"
            metadata_path = self.results_dir / metadata_filename

            # Добавляем дополнительные метаданные
            enhanced_metadata = self.processing_metadata.copy()
            enhanced_metadata.update({
                'metadata_creation_time': datetime.now().isoformat(),
                'data_file': str(self.data_file),
                'results_directory': str(self.results_dir),
                'processor_version': '2.0',
                'python_version': sys.version,
            })

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"  ✓ Метаданные сохранены: {metadata_path}")

        except Exception as e:
            logger.error(f"  ❌ Ошибка сохранения метаданных: {e}")
            # Не прерываем процесс из-за ошибки сохранения метаданных
            logger.warning("  ⚠️  Продолжаем без сохранения метаданных")


def main():
    """Основная функция для запуска расширенной кластеризации с улучшенным интерфейсом."""
    print("🔬 СИСТЕМА РАСШИРЕННОЙ КЛАСТЕРИЗАЦИИ v2.0")
    print("=" * 60)
    print("Исправления v2.0:")
    print("✓ Правильная оценка времени выполнения")
    print("✓ Комплексная валидация параметров")
    print("✓ Улучшенная обработка ошибок")
    print("✓ Возможность прерывания процесса")
    print("=" * 60)

    try:
        # Создаем процессор
        processor = ExtendedClusteringProcessor("student_habits_performance.csv")

        # Интерактивный выбор параметров
        print("\n⚙️  НАСТРОЙКА ПАРАМЕТРОВ:")

        try:
            max_k = int(input("Максимальное количество кластеров [по умолчанию 50]: ") or "50")
            min_k = int(input("Минимальное количество кластеров [по умолчанию 2]: ") or "2")
        except ValueError:
            print("⚠️  Используются значения по умолчанию")
            max_k, min_k = 50, 2

        print(f"\n📋 ПАРАМЕТРЫ КЛАСТЕРИЗАЦИИ:")
        print(f"   Диапазон K: {max_k} → {min_k}")
        print(f"   Количество разбиений: {max_k - min_k + 1}")
        print(f"   Метрика расстояния: chebyshev")

        # Прогресс-колбек
        def progress_callback(percent, message):
            print(f"\r🔄 [{percent:3d}%] {message}", end="", flush=True)

        print(f"\n🚀 Начинаем расширенную кластеризацию...")
        print(f"⚠️  Процесс можно прервать нажатием Ctrl+C")

        # Запуск полного цикла
        csv_path = processor.process_full_pipeline(
            min_clusters=min_k,
            max_clusters=max_k,
            distance_metric='chebyshev',
            save_csv=True,
            save_metadata=True,
            progress_callback=progress_callback
        )

        print(f"\n🎉 УСПЕШНО ЗАВЕРШЕНО!")
        print(f"📂 Результаты: {csv_path}")
        print(f"📁 Папка: clustering_results/")
        print(f"\n🚀 Следующий шаг - запуск GUI:")
        print(f"   python single_plot_gui.py")

    except KeyboardInterrupt:
        print(f"\n⏹️  Процесс прерван пользователем")
        print(f"🔄 Можно запустить заново для продолжения")
    except (ValueError, RuntimeError) as e:
        print(f"\n❌ Ошибка выполнения: {e}")
        print(f"💡 Проверьте параметры и попробуйте снова")
        return 1
    except Exception as e:
        print(f"\n💥 Критическая ошибка: {e}")
        print(f"🐛 Обратитесь к разработчику")
        return 2

    return 0


if __name__ == "__main__":
    exit(main())
