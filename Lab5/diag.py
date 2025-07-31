"""
ДИАГНОСТИКА ПРОБЛЕМ КЛАСТЕРИЗАЦИИ НА РЕАЛЬНЫХ ДАННЫХ
Специальный инструмент для анализа student_habits_performance.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def diagnose_clustering_issues(csv_file="student_habits_performance.csv"):
    """
    Полная диагностика проблем кластеризации на ваших данных.
    """
    print("🔬 ДИАГНОСТИКА ПРОБЛЕМ КЛАСТЕРИЗАЦИИ")
    print("=" * 50)

    # Загрузка данных
    try:
        data = pd.read_csv(csv_file)
        print(f"✅ Данные загружены: {data.shape}")
    except FileNotFoundError:
        print(f"❌ Файл {csv_file} не найден!")
        return

    # Выбираем только числовые колонки для анализа
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'student_id' in numeric_columns:
        numeric_columns.remove('student_id')

    print(f"📊 Числовые признаки: {len(numeric_columns)}")
    print(f"   {numeric_columns}")

    if len(numeric_columns) < 2:
        print("❌ Недостаточно числовых признаков для анализа")
        return

    # Берем первые 5 признаков для анализа
    analysis_features = numeric_columns[:5]
    X_original = data[analysis_features].copy()

    # Проверка на пропущенные значения
    print(f"\n🔍 ПРОВЕРКА КАЧЕСТВА ДАННЫХ:")
    print("-" * 30)

    missing_values = X_original.isnull().sum()
    if missing_values.sum() > 0:
        print(f"⚠️  Пропущенные значения:")
        for col, count in missing_values.items():
            if count > 0:
                print(f"   {col}: {count}")
        X_original = X_original.fillna(X_original.mean())
    else:
        print("✅ Пропущенных значений нет")

    # Анализ масштабов данных
    print(f"\n📏 АНАЛИЗ МАСШТАБОВ ДАННЫХ:")
    print("-" * 30)

    scales_problematic = False
    for col in analysis_features:
        min_val = X_original[col].min()
        max_val = X_original[col].max()
        range_val = max_val - min_val
        std_val = X_original[col].std()

        print(f"{col}:")
        print(f"   Диапазон: [{min_val:.2f}, {max_val:.2f}] (размах: {range_val:.2f})")
        print(f"   Стд. откл.: {std_val:.2f}")

        if range_val > 1000 or std_val > 100:
            print(f"   ⚠️  БОЛЬШОЙ МАСШТАБ - может вызывать проблемы!")
            scales_problematic = True

    if scales_problematic:
        print("\n🔴 ОБНАРУЖЕНА ПРОБЛЕМА: Разные масштабы данных!")
        print("   Это основная причина эффекта цепочки в односвязывающем методе")

    # Стандартизация данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_original)
    X_scaled_df = pd.DataFrame(X_scaled, columns=analysis_features)

    print(f"\n✅ Данные стандартизированы")

    # Тестирование кластеризации
    print(f"\n🧪 ТЕСТИРОВАНИЕ МЕТОДОВ КЛАСТЕРИЗАЦИИ:")
    print("-" * 40)

    test_clusters = [3, 4, 5]

    for n_clusters in test_clusters:
        print(f"\n🎯 Тест для K={n_clusters}:")

        # Наш односвязывающий алгоритм на исходных данных
        print("   📍 Односвязывающий (исходные данные):")
        try:
            from mathematical_algorithms import single_linkage_clustering

            labels_our_orig = single_linkage_clustering(
                X_original.values, n_clusters=n_clusters,
                metric='euclidean', return_hierarchy=False, verbose=False
            )

            sizes_our_orig = [np.sum(labels_our_orig == i) for i in range(n_clusters)]
            ratio_our_orig = max(sizes_our_orig) / min(sizes_our_orig) if min(sizes_our_orig) > 0 else float('inf')

            print(f"      Размеры: {sizes_our_orig}")
            print(f"      Соотношение: {ratio_our_orig:.2f}")

            if ratio_our_orig > 10:
                print(f"      🔴 ЭФФЕКТ ЦЕПОЧКИ!")
            else:
                print(f"      ✅ Сбалансированно")

        except ImportError:
            print("      ❌ Алгоритм недоступен")
            labels_our_orig = None

        # Наш односвязывающий алгоритм на стандартизированных данных
        print("   📍 Односвязывающий (стандартизированные данные):")
        try:
            labels_our_scaled = single_linkage_clustering(
                X_scaled, n_clusters=n_clusters,
                metric='euclidean', return_hierarchy=False, verbose=False
            )

            sizes_our_scaled = [np.sum(labels_our_scaled == i) for i in range(n_clusters)]
            ratio_our_scaled = max(sizes_our_scaled) / min(sizes_our_scaled) if min(sizes_our_scaled) > 0 else float('inf')

            print(f"      Размеры: {sizes_our_scaled}")
            print(f"      Соотношение: {ratio_our_scaled:.2f}")

            if ratio_our_scaled > 10:
                print(f"      🔴 Всё ещё эффект цепочки")
            else:
                print(f"      ✅ Стандартизация помогла!")

        except ImportError:
            labels_our_scaled = None

        # Эталонный sklearn односвязывающий
        print("   📍 Sklearn односвязывающий:")
        sklearn_single = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
        labels_sklearn = sklearn_single.fit_predict(X_scaled)

        sizes_sklearn = [np.sum(labels_sklearn == i) for i in range(n_clusters)]
        ratio_sklearn = max(sizes_sklearn) / min(sizes_sklearn) if min(sizes_sklearn) > 0 else float('inf')

        print(f"      Размеры: {sizes_sklearn}")
        print(f"      Соотношение: {ratio_sklearn:.2f}")

        if ratio_sklearn > 10:
            print(f"      🔴 Эффект цепочки и в sklearn!")
        else:
            print(f"      ✅ Sklearn работает лучше")

        # K-Means для сравнения
        print("   📍 K-Means:")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels_kmeans = kmeans.fit_predict(X_scaled)

        sizes_kmeans = [np.sum(labels_kmeans == i) for i in range(n_clusters)]
        ratio_kmeans = max(sizes_kmeans) / min(sizes_kmeans) if min(sizes_kmeans) > 0 else float('inf')

        print(f"      Размеры: {sizes_kmeans}")
        print(f"      Соотношение: {ratio_kmeans:.2f}")
        print(f"      ✅ K-Means обычно более сбалансирован")

    # Анализ структуры данных
    print(f"\n🔍 АНАЛИЗ СТРУКТУРЫ ДАННЫХ:")
    print("-" * 30)

    # Корреляционная матрица
    corr_matrix = X_scaled_df.corr()
    high_corr_pairs = []

    for i in range(len(analysis_features)):
        for j in range(i+1, len(analysis_features)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.7:
                high_corr_pairs.append((analysis_features[i], analysis_features[j], corr_val))

    if high_corr_pairs:
        print("⚠️  Высокие корреляции между признаками:")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"   {feat1} - {feat2}: {corr:.3f}")
    else:
        print("✅ Нет сильных корреляций между признаками")

    # Поиск выбросов
    print(f"\n🎯 ПОИСК ВЫБРОСОВ:")
    print("-" * 20)

    outliers_found = False
    for col in analysis_features:
        Q1 = X_original[col].quantile(0.25)
        Q3 = X_original[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = X_original[(X_original[col] < lower_bound) | (X_original[col] > upper_bound)]

        if len(outliers) > 0:
            print(f"{col}: {len(outliers)} выбросов ({len(outliers)/len(X_original)*100:.1f}%)")
            outliers_found = True

    if not outliers_found:
        print("✅ Значительных выбросов не обнаружено")

    return X_original, X_scaled_df


def create_visualization_comparison(X_original, X_scaled):
    """
    Создание визуализации для сравнения результатов кластеризации.
    """
    print(f"\n📊 СОЗДАНИЕ ВИЗУАЛИЗАЦИИ...")

    # Берем первые 2 признака для 2D визуализации
    X_orig_2d = X_original.iloc[:, :2].values
    X_scaled_2d = X_scaled.iloc[:, :2].values

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Сравнение методов кластеризации', fontsize=16)

    n_clusters = 4

    # Исходные данные
    axes[0, 0].scatter(X_orig_2d[:, 0], X_orig_2d[:, 1], alpha=0.6)
    axes[0, 0].set_title('Исходные данные')
    axes[0, 0].grid(True, alpha=0.3)

    # Стандартизированные данные
    axes[1, 0].scatter(X_scaled_2d[:, 0], X_scaled_2d[:, 1], alpha=0.6)
    axes[1, 0].set_title('Стандартизированные данные')
    axes[1, 0].grid(True, alpha=0.3)

    # Односвязывающий на исходных данных
    try:
        from mathematical_algorithms import single_linkage_clustering
        labels_orig = single_linkage_clustering(X_orig_2d, n_clusters=n_clusters,
                                               return_hierarchy=False, verbose=False)
        axes[0, 1].scatter(X_orig_2d[:, 0], X_orig_2d[:, 1], c=labels_orig, alpha=0.6)
        axes[0, 1].set_title('Односвязывающий (исходные)')

        sizes = [np.sum(labels_orig == i) for i in range(n_clusters)]
        axes[0, 1].text(0.02, 0.98, f'Размеры: {sizes}', transform=axes[0, 1].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white'))
    except:
        axes[0, 1].text(0.5, 0.5, 'Алгоритм\nнедоступен', ha='center', va='center',
                       transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Односвязывающий (недоступен)')

    # Односвязывающий на стандартизированных данных
    try:
        labels_scaled = single_linkage_clustering(X_scaled_2d, n_clusters=n_clusters,
                                                 return_hierarchy=False, verbose=False)
        axes[1, 1].scatter(X_scaled_2d[:, 0], X_scaled_2d[:, 1], c=labels_scaled, alpha=0.6)
        axes[1, 1].set_title('Односвязывающий (стандарт.)')

        sizes = [np.sum(labels_scaled == i) for i in range(n_clusters)]
        axes[1, 1].text(0.02, 0.98, f'Размеры: {sizes}', transform=axes[1, 1].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white'))
    except:
        axes[1, 1].text(0.5, 0.5, 'Алгоритм\nнедоступен', ha='center', va='center',
                       transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Односвязывающий (недоступен)')

    # K-Means
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_scaled_2d)

    axes[0, 2].scatter(X_scaled_2d[:, 0], X_scaled_2d[:, 1], c=labels_kmeans, alpha=0.6)
    axes[0, 2].set_title('K-Means')

    sizes_kmeans = [np.sum(labels_kmeans == i) for i in range(n_clusters)]
    axes[0, 2].text(0.02, 0.98, f'Размеры: {sizes_kmeans}', transform=axes[0, 2].transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white'))

    # Sklearn односвязывающий
    sklearn_single = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
    labels_sklearn = sklearn_single.fit_predict(X_scaled_2d)

    axes[1, 2].scatter(X_scaled_2d[:, 0], X_scaled_2d[:, 1], c=labels_sklearn, alpha=0.6)
    axes[1, 2].set_title('Sklearn односвязывающий')

    sizes_sklearn = [np.sum(labels_sklearn == i) for i in range(n_clusters)]
    axes[1, 2].text(0.02, 0.98, f'Размеры: {sizes_sklearn}', transform=axes[1, 2].transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white'))

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def generate_recommendations(X_original, X_scaled):
    """
    Генерация конкретных рекомендаций для улучшения кластеризации.
    """
    print(f"\n💡 ПЕРСОНАЛЬНЫЕ РЕКОМЕНДАЦИИ:")
    print("=" * 40)

    # Проверяем улучшение от стандартизации
    try:
        from mathematical_algorithms import single_linkage_clustering

        labels_orig = single_linkage_clustering(X_original.values[:, :2], n_clusters=4,
                                               return_hierarchy=False, verbose=False)
        labels_scaled = single_linkage_clustering(X_scaled.values[:, :2], n_clusters=4,
                                                 return_hierarchy=False, verbose=False)

        sizes_orig = [np.sum(labels_orig == i) for i in range(4)]
        sizes_scaled = [np.sum(labels_scaled == i) for i in range(4)]

        ratio_orig = max(sizes_orig) / min(sizes_orig) if min(sizes_orig) > 0 else float('inf')
        ratio_scaled = max(sizes_scaled) / min(sizes_scaled) if min(sizes_scaled) > 0 else float('inf')

        print(f"🔍 АНАЛИЗ РЕЗУЛЬТАТОВ:")
        print(f"   Без стандартизации: соотношение {ratio_orig:.2f}")
        print(f"   Со стандартизацией: соотношение {ratio_scaled:.2f}")

        if ratio_scaled < ratio_orig * 0.8:
            print(f"✅ СТАНДАРТИЗАЦИЯ ПОМОГАЕТ! Используйте её в GUI")
        else:
            print(f"❌ Стандартизация не решает проблему")

        if ratio_scaled > 5:
            print(f"\n🎯 ОСНОВНЫЕ РЕКОМЕНДАЦИИ:")
            print(f"1. 🔄 ПЕРЕКЛЮЧИТЕСЬ НА K-MEANS в GUI")
            print(f"2. 📊 Увеличьте количество кластеров (попробуйте K=6-8)")
            print(f"3. 🧹 Очистите данные от выбросов")
            print(f"4. 📐 Попробуйте метрику 'chebyshev' вместо 'euclidean'")
        else:
            print(f"\n✅ Односвязывающий метод работает приемлемо")

    except ImportError:
        print(f"❌ Не удалось протестировать наш алгоритм")

    print(f"\n🎮 ПРАКТИЧЕСКИЕ ШАГИ В GUI:")
    print(f"1. Убедитесь что используется StandardScaler в предобработке")
    print(f"2. Попробуйте K-Means вместо односвязывающего метода")
    print(f"3. Если нужен именно односвязывающий - увеличьте K до 6-8")
    print(f"4. Используйте метрику 'chebyshev' для односвязывающего")
    print(f"5. Попробуйте выбрать другие признаки через СПА")


def main():
    """Основная функция диагностики."""
    print("🚀 ЗАПУСК ДИАГНОСТИКИ ДЛЯ STUDENT_HABITS_PERFORMANCE.CSV")
    print("=" * 60)

    try:
        X_orig, X_scaled = diagnose_clustering_issues()

        if X_orig is not None:
            create_visualization_comparison(X_orig, X_scaled)
            generate_recommendations(X_orig, X_scaled)

    except Exception as e:
        print(f"❌ Ошибка в диагностике: {e}")

    print(f"\n🎯 ЗАКЛЮЧЕНИЕ:")
    print("Эффект цепочки в односвязывающем методе - это часто")
    print("естественное поведение на реальных данных, а не ошибка кода!")


if __name__ == "__main__":
    main()
