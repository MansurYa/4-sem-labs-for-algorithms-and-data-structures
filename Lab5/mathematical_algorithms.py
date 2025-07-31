import numpy as np
from typing import Union, Literal, Tuple, List, Optional, Dict, Any
import warnings
from collections import defaultdict


def validate_vectors(x: np.ndarray, y: np.ndarray, function_name: str) -> None:
    """
    Комплексная валидация входных векторов для метрик расстояний.

    :param x, y: Векторы для проверки
    :param function_name: Имя функции для информативных сообщений об ошибках
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError(f"{function_name}: входные данные должны быть numpy массивами")

    if x.shape != y.shape:
        raise ValueError(f"{function_name}: векторы должны иметь одинаковую размерность: {x.shape} != {y.shape}")

    if len(x.shape) != 1:
        raise ValueError(f"{function_name}: ожидаются одномерные векторы, получены: {x.shape}")

    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError(f"{function_name}: входные данные содержат NaN значения")

    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError(f"{function_name}: входные данные содержат бесконечные значения")

    if len(x) == 0:
        raise ValueError(f"{function_name}: векторы не могут быть пустыми")


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Вычисление евклидова расстояния с повышенной численной стабильностью.

    Математическое определение: d(x, y) = √(Σ(xi - yi)²) = ||x - y||₂

    :param x, y: Векторы одинаковой размерности
    :return: евклидово расстояние

    Example:
    >>> euclidean_distance(np.array([1, 2, 3]), np.array([4, 5, 6]))
    5.196152422706632
    """
    validate_vectors(x, y, "euclidean_distance")
    return float(np.linalg.norm(x - y, ord=2))


def euclidean_distance_squared(x: np.ndarray, y: np.ndarray) -> float:
    """
    Вычисление квадрата евклидова расстояния.

    Математическое определение: d²(x, y) = Σ(xi - yi)² = ||x - y||₂²

    Преимущество: избегает вычисления квадратного корня, повышая эффективность
    при сохранении порядка расстояний для целей кластеризации.
    """
    validate_vectors(x, y, "euclidean_distance_squared")
    diff = x - y
    return float(np.dot(diff, diff))


def chebyshev_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Вычисление расстояния Чебышева с обработкой граничных случаев.

    Математическое определение: d(x, y) = max|xi - yi| = ||x - y||∞
    """
    validate_vectors(x, y, "chebyshev_distance")
    return float(np.max(np.abs(x - y)))


def compute_distance_matrix(data: np.ndarray,
                          metric: Literal['euclidean', 'euclidean_squared', 'chebyshev'] = 'euclidean') -> np.ndarray:
    """
    Эффективное и робастное вычисление матрицы расстояний.

    :param data: Матрица данных, shape (n_samples, n_features)
    :param metric: Используемая метрика расстояния
    :return: Симметричная матрица расстояний, shape (n_samples, n_samples)
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Данные должны быть numpy массивом")

    if len(data.shape) != 2:
        raise ValueError(f"Ожидается двумерная матрица, получена: {data.shape}")

    if data.shape[0] < 2:
        raise ValueError("Необходимо минимум 2 точки для вычисления расстояний")

    if np.any(np.isnan(data)):
        raise ValueError("Данные содержат NaN значения")

    if np.any(np.isinf(data)):
        raise ValueError("Данные содержат бесконечные значения")

    n_samples = data.shape[0]

    distance_functions = {
        'euclidean': euclidean_distance,
        'euclidean_squared': euclidean_distance_squared,
        'chebyshev': chebyshev_distance
    }

    if metric not in distance_functions:
        raise ValueError(f"Неподдерживаемая метрика: {metric}. "
                        f"Доступные: {list(distance_functions.keys())}")

    distance_func = distance_functions[metric]
    distance_matrix = np.zeros((n_samples, n_samples), dtype=np.float64)

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            try:
                dist = distance_func(data[i], data[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
            except Exception as e:
                raise RuntimeError(f"Ошибка вычисления расстояния между точками {i} и {j}: {e}")

    return distance_matrix


def single_linkage_clustering(data: np.ndarray,
                            n_clusters: int,
                            metric: str = 'euclidean',
                            return_hierarchy: bool = True,
                            verbose: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    """
    Робустная реализация односвязывающего метода с комплексной защитой от ошибок.

    :param data: Входные данные для кластеризации, shape (n_samples, n_features)
    :param n_clusters: Целевое количество кластеров
    :param metric: Метрика расстояния
    :param return_hierarchy: Возвращать ли иерархию объединений
    :param verbose: Выводить ли прогресс выполнения
    :return: labels или (labels, hierarchy)
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Данные должны быть numpy массивом")

    if len(data.shape) != 2:
        raise ValueError(f"Ожидается двумерная матрица, получена: {data.shape}")

    n_samples, n_features = data.shape

    if n_samples < 2:
        raise ValueError("Необходимо минимум 2 точки для кластеризации")

    if n_clusters < 1:
        raise ValueError("Количество кластеров должно быть положительным")

    if n_clusters > n_samples:
        raise ValueError(f"Количество кластеров ({n_clusters}) не может превышать "
                        f"количество точек ({n_samples})")

    if n_clusters == n_samples:
        if verbose:
            print("Каждая точка образует отдельный кластер")
        labels = np.arange(n_samples)
        return (labels, []) if return_hierarchy else labels

    if verbose:
        print(f"Вычисление матрицы расстояний для {n_samples} точек...")

    try:
        distance_matrix = compute_distance_matrix(data, metric)
    except Exception as e:
        raise RuntimeError(f"Не удалось вычислить матрицу расстояний: {e}")

    finite_distances = distance_matrix[np.triu_indices(n_samples, k=1)]

    if len(finite_distances) == 0:
        raise RuntimeError("Не удалось найти конечные расстояния между точками")

    if np.all(finite_distances == 0):
        warnings.warn("Все точки идентичны, результат кластеризации может быть неопределенным")

    if np.any(np.isinf(finite_distances)):
        raise ValueError("Обнаружены бесконечные расстояния между точками")

    clusters = [set([i]) for i in range(n_samples)]
    hierarchy = []

    if verbose:
        print(f"Начальное количество кластеров: {len(clusters)}")

    iteration = 0
    while len(clusters) > n_clusters:
        iteration += 1

        if verbose and (iteration % 50 == 0 or iteration <= 10):
            print(f"Итерация {iteration}: {len(clusters)} кластеров")

        min_distance = float('inf')
        merge_indices = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster_distance = float('inf')

                for point_i in clusters[i]:
                    for point_j in clusters[j]:
                        point_distance = distance_matrix[point_i, point_j]
                        if point_distance < cluster_distance:
                            cluster_distance = point_distance

                if cluster_distance < min_distance:
                    min_distance = cluster_distance
                    merge_indices = (i, j)

        if merge_indices is None:
            raise RuntimeError(f"Не удалось найти кластеры для объединения на итерации {iteration}")

        if not np.isfinite(min_distance):
            raise RuntimeError(f"Получено неконечное расстояние между кластерами: {min_distance}")

        i, j = merge_indices

        if verbose and len(clusters) <= 10:
            print(f"  Объединяются кластеры {i} (размер {len(clusters[i])}) "
                  f"и {j} (размер {len(clusters[j])}), расстояние: {min_distance:.6f}")

        merged_cluster = clusters[i].union(clusters[j])

        if return_hierarchy:
            hierarchy.append({
                'merged_clusters': (i, j),
                'cluster_sizes': (len(clusters[i]), len(clusters[j])),
                'new_cluster_size': len(merged_cluster),
                'distance': min_distance,
                'iteration': iteration,
                'remaining_clusters': len(clusters) - 1
            })

        new_clusters = []
        for idx in range(len(clusters)):
            if idx != i and idx != j:
                new_clusters.append(clusters[idx])
        new_clusters.append(merged_cluster)

        clusters = new_clusters

    labels = np.zeros(n_samples, dtype=int)
    for cluster_id, cluster_points in enumerate(clusters):
        for point_id in cluster_points:
            labels[point_id] = cluster_id

    if verbose:
        print(f"Кластеризация завершена после {iteration} итераций.")
        print(f"Размеры финальных кластеров: {[len(cluster) for cluster in clusters]}")

    return (labels, hierarchy) if return_hierarchy else labels


def spa_feature_selection(data: np.ndarray,
                         n_features: int,
                         quality_function,
                         max_iterations: int = 1000,
                         learning_rate: float = 0.05,
                         min_learning_rate: float = 0.0001,
                         convergence_threshold: float = 1e-6,
                         early_stopping_patience: int = 20,
                         random_state: int = 42,
                         verbose: bool = False) -> Tuple[List[int], float, List[dict]]:
    """
    Численно стабильная реализация алгоритма СПА с улучшенной сходимостью.

    :param data: Исходные данные, shape (n_samples, n_features)
    :param n_features: Количество признаков для выбора
    :param quality_function: Функция оценки качества кластеризации
    :param max_iterations: Максимальное количество итераций
    :param learning_rate: Начальный коэффициент обучения
    :param min_learning_rate: Минимальный коэффициент обучения
    :param convergence_threshold: Порог сходимости для ранней остановки
    :param early_stopping_patience: Количество итераций без улучшения
    :param random_state: Seed для воспроизводимости
    :param verbose: Выводить ли прогресс выполнения
    :return: (best_features, best_quality, convergence_history)
    """
    np.random.seed(random_state)

    if not isinstance(data, np.ndarray):
        raise TypeError("Данные должны быть numpy массивом")

    if len(data.shape) != 2:
        raise ValueError(f"Ожидается двумерная матрица, получена: {data.shape}")

    n_samples, n_total_features = data.shape

    if n_samples < 2:
        raise ValueError("Необходимо минимум 2 объекта для анализа")

    if n_features < 1:
        raise ValueError("Количество выбираемых признаков должно быть положительным")

    if n_features >= n_total_features:
        if verbose:
            print(f"Запрошено {n_features} признаков из {n_total_features} доступных. "
                  f"Возвращаются все признаки.")
        return list(range(n_total_features)), 1.0, []

    if not callable(quality_function):
        raise TypeError("quality_function должна быть вызываемой функцией")

    if not 0 < learning_rate <= 1:
        raise ValueError("learning_rate должен быть в диапазоне (0, 1]")

    if not 0 < min_learning_rate <= learning_rate:
        raise ValueError("min_learning_rate должен быть в диапазоне (0, learning_rate]")

    feature_probabilities = np.ones(n_total_features, dtype=np.float64) / n_total_features

    quality_history = []
    best_quality = -float('inf')
    best_features = None
    convergence_history = []
    iterations_without_improvement = 0

    if verbose:
        print(f"Запуск СПА: {n_total_features} признаков → {n_features} признаков")
        print(f"Параметры: lr={learning_rate}, max_iter={max_iterations}")

    for iteration in range(max_iterations):
        current_lr = max(learning_rate * (0.95 ** (iteration // 10)), min_learning_rate)

        random_values = np.random.uniform(0, 1, n_total_features)
        selected_mask = random_values < feature_probabilities
        selected_features = np.where(selected_mask)[0].tolist()

        if len(selected_features) != n_features:
            if len(selected_features) < n_features:
                remaining_features = [i for i in range(n_total_features)
                                    if i not in selected_features]
                if remaining_features:
                    additional_count = min(n_features - len(selected_features),
                                         len(remaining_features))

                    remaining_probs = feature_probabilities[remaining_features]

                    if np.sum(remaining_probs) > 0:
                        try:
                            additional_features = np.random.choice(
                                remaining_features,
                                additional_count,
                                replace=False,
                                p=remaining_probs / np.sum(remaining_probs)
                            )
                            selected_features.extend(additional_features.tolist())
                        except ValueError:
                            additional_features = np.random.choice(
                                remaining_features,
                                additional_count,
                                replace=False
                            )
                            selected_features.extend(additional_features.tolist())
                    else:
                        additional_features = np.random.choice(
                            remaining_features,
                            additional_count,
                            replace=False
                        )
                        selected_features.extend(additional_features.tolist())
            else:
                selected_probs = [(idx, feature_probabilities[idx])
                                for idx in selected_features]
                selected_probs.sort(key=lambda x: x[1], reverse=True)
                selected_features = [idx for idx, _ in selected_probs[:n_features]]

        if len(selected_features) != n_features:
            selected_features = np.random.choice(
                range(n_total_features), n_features, replace=False
            ).tolist()

        try:
            selected_data = data[:, selected_features]
            current_quality = quality_function(selected_data)

            if not np.isfinite(current_quality):
                if verbose:
                    print(f"Итерация {iteration+1}: некорректное качество, пропускаем")
                current_quality = 0.0

            quality_history.append(current_quality)

            if current_quality > best_quality:
                best_quality = current_quality
                best_features = selected_features.copy()
                iterations_without_improvement = 0
                if verbose:
                    print(f"Итерация {iteration+1}: новое лучшее качество = {best_quality:.6f}")
            else:
                iterations_without_improvement += 1

        except Exception as e:
            if verbose:
                print(f"Ошибка в оценке качества на итерации {iteration+1}: {e}")
            current_quality = 0.0
            quality_history.append(current_quality)
            iterations_without_improvement += 1

        if len(quality_history) >= 2:
            mean_quality = np.mean(quality_history)
            quality_deviation = current_quality - mean_quality

            quality_std = np.std(quality_history)
            quality_std = max(quality_std, 1e-6)

            normalized_deviation = quality_deviation / quality_std
            normalized_deviation = np.clip(normalized_deviation, -5.0, 5.0)

            for i in range(n_total_features):
                if i in selected_features:
                    feature_probabilities[i] += current_lr * normalized_deviation
                else:
                    remaining_count = n_total_features - len(selected_features)
                    if remaining_count > 0:
                        feature_probabilities[i] -= (current_lr * normalized_deviation) / remaining_count

            feature_probabilities = np.maximum(feature_probabilities, 1e-8)
            prob_sum = np.sum(feature_probabilities)

            if prob_sum > 1e-10:
                feature_probabilities /= prob_sum
            else:
                feature_probabilities = np.ones(n_total_features) / n_total_features
                if verbose:
                    print(f"Итерация {iteration+1}: аварийное восстановление вероятностей")

        convergence_info = {
            'iteration': iteration + 1,
            'current_quality': current_quality,
            'best_quality': best_quality,
            'learning_rate': current_lr,
            'selected_features': selected_features.copy(),
            'feature_probabilities': feature_probabilities.copy(),
            'quality_std': quality_std if len(quality_history) >= 2 else 0.0
        }
        convergence_history.append(convergence_info)

        if iterations_without_improvement >= early_stopping_patience:
            if verbose:
                print(f"Ранняя остановка на итерации {iteration+1}: "
                      f"{early_stopping_patience} итераций без улучшения")
            break

        if len(quality_history) >= 10:
            recent_qualities = quality_history[-10:]
            if np.std(recent_qualities) < convergence_threshold:
                if verbose:
                    print(f"Ранняя остановка на итерации {iteration+1}: достигнута сходимость")
                break

    if best_features is None:
        best_features = np.random.choice(range(n_total_features), n_features, replace=False).tolist()
        best_quality = 0.0
        if verbose:
            print("Предупреждение: использованы случайные признаки из-за отсутствия валидных результатов")

    if verbose:
        print(f"СПА завершен. Лучшее качество: {best_quality:.6f}")
        print(f"Выбранные признаки: {sorted(best_features)}")

    return best_features, best_quality, convergence_history


def compute_binomial_coefficient(n: int, k: int = 2) -> int:
    """
    Эффективное и читаемое вычисление биномиального коэффициента C(n,k).

    Специализировано для k=2: C(n,2) = n*(n-1)/2

    :param n: Количество элементов
    :param k: Размер выборки (по умолчанию 2 для подсчета пар)
    :return: биномиальный коэффициент
    """
    if n < k or n < 0 or k < 0:
        return 0

    if k == 0 or k == n:
        return 1

    if k == 2:
        return n * (n - 1) // 2

    if k > n - k:
        k = n - k

    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)

    return result


def fowlkes_mallows_index(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Численно стабильное вычисление индекса Фоулкса-Мэллова.

    Математическое определение: FM = √(Precision × Recall)

    :param true_labels: Истинные метки кластеров
    :param pred_labels: Предсказанные метки кластеров
    :return: индекс Фоулкса-Мэллова в диапазоне [0, 1]
    """
    true_labels = np.asarray(true_labels, dtype=int)
    pred_labels = np.asarray(pred_labels, dtype=int)

    if len(true_labels) != len(pred_labels):
        raise ValueError(f"Длины массивов меток должны совпадать: "
                        f"{len(true_labels)} != {len(pred_labels)}")

    n_samples = len(true_labels)

    if n_samples < 2:
        return 1.0

    if n_samples == 2:
        return 1.0 if (true_labels[0] == true_labels[1]) == (pred_labels[0] == pred_labels[1]) else 0.0

    unique_true = np.unique(true_labels)
    unique_pred = np.unique(pred_labels)

    contingency_matrix = np.zeros((len(unique_true), len(unique_pred)), dtype=int)

    true_to_idx = {label: idx for idx, label in enumerate(unique_true)}
    pred_to_idx = {label: idx for idx, label in enumerate(unique_pred)}

    for true_label, pred_label in zip(true_labels, pred_labels):
        i = true_to_idx[true_label]
        j = pred_to_idx[pred_label]
        contingency_matrix[i, j] += 1

    tp = 0
    for i in range(contingency_matrix.shape[0]):
        for j in range(contingency_matrix.shape[1]):
            n_ij = contingency_matrix[i, j]
            tp += compute_binomial_coefficient(n_ij, 2)

    true_cluster_sizes = np.bincount(true_labels)
    pred_cluster_sizes = np.bincount(pred_labels)

    predicted_pairs = sum(compute_binomial_coefficient(size, 2) for size in pred_cluster_sizes)
    true_pairs = sum(compute_binomial_coefficient(size, 2) for size in true_cluster_sizes)

    if predicted_pairs == 0 and true_pairs == 0:
        return 1.0

    if predicted_pairs == 0 or true_pairs == 0:
        return 0.0

    precision = tp / predicted_pairs if predicted_pairs > 0 else 0.0
    recall = tp / true_pairs if true_pairs > 0 else 0.0

    if precision * recall <= 0:
        return 0.0

    fm_index = np.sqrt(precision * recall)

    if not (0 <= fm_index <= 1):
        raise RuntimeError(f"Некорректное значение индекса Фоулкса-Мэллова: {fm_index}. "
                          f"Precision: {precision}, Recall: {recall}")

    return float(fm_index)


def single_linkage_clustering_with_history(data: np.ndarray,
                                         min_clusters: int = 2,
                                         max_clusters: int = 10,
                                         metric: str = 'euclidean',
                                         verbose: bool = False) -> Tuple[Dict[int, np.ndarray], List[Dict]]:
    """
    Робустная реализация односвязывающего метода с сохранением полной истории кластеризации.

    Выполняет кластеризацию от начального состояния (каждая точка - отдельный кластер)
    до min_clusters, сохраняя все промежуточные разбиения.

    :param data: Входные данные для кластеризации, shape (n_samples, n_features)
    :param min_clusters: Минимальное количество кластеров (финальное состояние)
    :param max_clusters: Максимальное количество кластеров для сохранения
    :param metric: Метрика расстояния
    :param verbose: Выводить ли прогресс выполнения
    :return: (история_разбиений, детальная_история)
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Данные должны быть numpy массивом")

    if len(data.shape) != 2:
        raise ValueError(f"Ожидается двумерная матрица, получена: {data.shape}")

    n_samples, n_features = data.shape

    if n_samples < 2:
        raise ValueError("Необходимо минимум 2 точки для кластеризации")

    if min_clusters < 1:
        raise ValueError("Количество кластеров должно быть положительным")

    if max_clusters > n_samples:
        max_clusters = n_samples
        if verbose:
            print(f"Максимальное количество кластеров ограничено размером данных: {max_clusters}")

    if min_clusters > max_clusters:
        raise ValueError(f"min_clusters ({min_clusters}) не может быть больше max_clusters ({max_clusters})")

    if verbose:
        print(f"Запуск односвязывающей кластеризации с сохранением истории")
        print(f"Данные: {n_samples} точек, {n_features} признаков")
        print(f"Целевое количество кластеров: от {max_clusters} до {min_clusters}")
        print(f"Метрика: {metric}")

    try:
        distance_matrix = compute_distance_matrix(data, metric)
    except Exception as e:
        raise RuntimeError(f"Не удалось вычислить матрицу расстояний: {e}")

    # Проверка корректности матрицы расстояний
    finite_distances = distance_matrix[np.triu_indices(n_samples, k=1)]

    if len(finite_distances) == 0:
        raise RuntimeError("Не удалось найти конечные расстояния между точками")

    if np.any(np.isinf(finite_distances)):
        raise ValueError("Обнаружены бесконечные расстояния между точками")

    # ===== ОСНОВНОЙ АЛГОРИТМ КЛАСТЕРИЗАЦИИ =====

    # Инициализация: каждая точка - отдельный кластер
    clusters = [set([i]) for i in range(n_samples)]

    # Словари для сохранения истории
    clustering_history = {}  # {k: labels_array}
    detailed_history = []    # Детальная информация о каждом объединении

    if verbose:
        print(f"Начальное количество кластеров: {len(clusters)}")

    # Функция для создания массива меток из списка кластеров
    def create_labels_array(cluster_list):
        labels = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster_points in enumerate(cluster_list):
            for point_id in cluster_points:
                labels[point_id] = cluster_id
        return labels

    # Сохраняем начальные состояния (если нужно)
    current_k = len(clusters)
    if current_k <= max_clusters and current_k >= min_clusters:
        clustering_history[current_k] = create_labels_array(clusters)
        if verbose:
            print(f"Сохранено разбиение на {current_k} кластеров")

    iteration = 0
    while len(clusters) > min_clusters:
        iteration += 1
        current_k = len(clusters)

        if verbose and (iteration % 50 == 0 or iteration <= 10):
            print(f"Итерация {iteration}: {current_k} кластеров")

        # Находим два ближайших кластера
        min_distance = float('inf')
        merge_indices = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Вычисляем минимальное расстояние между кластерами (single linkage)
                cluster_distance = float('inf')

                for point_i in clusters[i]:
                    for point_j in clusters[j]:
                        point_distance = distance_matrix[point_i, point_j]
                        if point_distance < cluster_distance:
                            cluster_distance = point_distance

                if cluster_distance < min_distance:
                    min_distance = cluster_distance
                    merge_indices = (i, j)

        if merge_indices is None:
            raise RuntimeError(f"Не удалось найти кластеры для объединения на итерации {iteration}")

        if not np.isfinite(min_distance):
            raise RuntimeError(f"Получено неконечное расстояние между кластерами: {min_distance}")

        # Выполняем объединение кластеров
        i, j = merge_indices

        if verbose and current_k <= 15:
            print(f"  Объединяются кластеры {i} (размер {len(clusters[i])}) "
                  f"и {j} (размер {len(clusters[j])}), расстояние: {min_distance:.6f}")

        merged_cluster = clusters[i].union(clusters[j])

        # Сохраняем детальную информацию об объединении
        merge_info = {
            'iteration': iteration,
            'merged_clusters': (i, j),
            'cluster_sizes': (len(clusters[i]), len(clusters[j])),
            'new_cluster_size': len(merged_cluster),
            'merge_distance': min_distance,
            'clusters_before': current_k,
            'clusters_after': current_k - 1
        }
        detailed_history.append(merge_info)

        # Создаем новый список кластеров
        new_clusters = []
        for idx in range(len(clusters)):
            if idx != i and idx != j:
                new_clusters.append(clusters[idx])
        new_clusters.append(merged_cluster)

        clusters = new_clusters
        new_k = len(clusters)

        # Сохраняем разбиение, если оно в нужном диапазоне
        if new_k <= max_clusters and new_k >= min_clusters:
            clustering_history[new_k] = create_labels_array(clusters)
            if verbose:
                print(f"  Сохранено разбиение на {new_k} кластеров")

    if verbose:
        print(f"Кластеризация завершена после {iteration} итераций.")
        print(f"Сохранено {len(clustering_history)} различных разбиений")
        saved_k_values = sorted(clustering_history.keys(), reverse=True)
        print(f"Доступные разбиения: K = {saved_k_values}")

    return clustering_history, detailed_history


def validate_clustering_history(clustering_history: Dict[int, np.ndarray],
                              data_shape: Tuple[int, int]) -> bool:
    """
    Валидация корректности сохраненной истории кластеризации.

    :param clustering_history: Словарь с историей разбиений
    :param data_shape: Размерность исходных данных
    :return: True если история корректна
    """
    n_samples = data_shape[0]

    try:
        for k, labels in clustering_history.items():
            # Проверка размерности
            if len(labels) != n_samples:
                print(f"Ошибка: неправильная размерность меток для K={k}: {len(labels)} != {n_samples}")
                return False

            # Проверка количества кластеров
            unique_labels = np.unique(labels)
            if len(unique_labels) != k:
                print(f"Ошибка: количество уникальных меток для K={k}: {len(unique_labels)} != {k}")
                return False

            # Проверка последовательности меток (должны быть 0, 1, 2, ..., k-1)
            expected_labels = set(range(k))
            actual_labels = set(unique_labels)
            if actual_labels != expected_labels:
                print(f"Ошибка: неправильные метки для K={k}: {actual_labels} != {expected_labels}")
                return False

        print(f"✅ Валидация истории кластеризации пройдена успешно")
        print(f"   Проверено {len(clustering_history)} разбиений")
        return True

    except Exception as e:
        print(f"❌ Ошибка при валидации истории кластеризации: {e}")
        return False


def get_cluster_statistics(clustering_history: Dict[int, np.ndarray],
                         data: np.ndarray) -> Dict[int, Dict[str, Any]]:
    """
    Вычисление статистики для каждого разбиения в истории кластеризации.

    :param clustering_history: История разбиений
    :param data: Исходные данные
    :return: Словарь со статистикой для каждого K
    """
    statistics = {}

    for k, labels in clustering_history.items():
        try:
            # Базовая статистика
            cluster_counts = np.bincount(labels)

            # Вычисляем центроиды кластеров
            centroids = []
            for cluster_id in range(k):
                cluster_mask = labels == cluster_id
                cluster_points = data[cluster_mask]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    centroids.append(centroid)
                else:
                    centroids.append(np.zeros(data.shape[1]))

            centroids = np.array(centroids)

            # Внутрикластерная дисперсия
            within_cluster_variance = 0.0
            for cluster_id in range(k):
                cluster_mask = labels == cluster_id
                cluster_points = data[cluster_mask]
                if len(cluster_points) > 0:
                    centroid = centroids[cluster_id]
                    within_cluster_variance += np.sum((cluster_points - centroid) ** 2)

            # Между кластерная дисперсия
            global_centroid = np.mean(data, axis=0)
            between_cluster_variance = 0.0
            for cluster_id in range(k):
                cluster_size = np.sum(labels == cluster_id)
                if cluster_size > 0:
                    centroid = centroids[cluster_id]
                    between_cluster_variance += cluster_size * np.sum((centroid - global_centroid) ** 2)

            # Коэффициент силуэта (если возможно)
            silhouette_avg = None
            if k > 1 and len(np.unique(labels)) > 1:
                try:
                    from sklearn.metrics import silhouette_score
                    silhouette_avg = silhouette_score(data, labels)
                except Exception:
                    silhouette_avg = None

            statistics[k] = {
                'cluster_counts': cluster_counts.tolist(),
                'min_cluster_size': int(cluster_counts.min()),
                'max_cluster_size': int(cluster_counts.max()),
                'avg_cluster_size': float(cluster_counts.mean()),
                'centroids': centroids.tolist(),
                'within_cluster_variance': float(within_cluster_variance),
                'between_cluster_variance': float(between_cluster_variance),
                'silhouette_score': float(silhouette_avg) if silhouette_avg is not None else None
            }

        except Exception as e:
            print(f"Ошибка при вычислении статистики для K={k}: {e}")
            statistics[k] = {'error': str(e)}

    return statistics
