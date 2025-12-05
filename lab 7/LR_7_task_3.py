import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib

# Завантаження вхідних даних
try:
    X = np.loadtxt('data_clustering.txt', delimiter=',')  # Завантаження даних з файлу
except OSError:
    print("Файл 'data_clustering.txt' не знайдено. Будь ласка, перевірте шлях до файлу або згенеруйте дані.")
    X = np.zeros((10, 2))  # Пустий масив для запобігання помилок

# Оцінка ширини вікна (bandwidth) для Mean Shift
bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))  # Підбір ширини вікна

# Навчання моделі кластеризації методом зсуву середнього
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

# Витяг центру кластерів
cluster_centers = meanshift_model.cluster_centers_
print('\nCenters of clusters:\n', cluster_centers)

# Оцінка кількості кластерів
labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print("\nNumber of clusters in input data =", num_clusters)
plt.figure(figsize=(10, 7))

# Використовуємо сучасний метод для отримання кольорової палітри
colors = matplotlib.colormaps['tab10'].resampled(num_clusters)

for i in range(num_clusters):
    # Відображення точок поточного кластера кольором та з прозорістю
    plt.scatter(X[labels == i, 0], X[labels == i, 1],
                marker='o',
                color=colors(i),
                s=100,           # розмір точок
                alpha=0.7,       # прозорість
                label=f'Кластер {i+1}',
                edgecolor='k')   # чорна обводка для кращої видимості

    # Відображення центру поточного кластера
    cluster_center = cluster_centers[i]
    plt.scatter(cluster_center[0], cluster_center[1],
                marker='X',
                color='red',
                s=250,           # великий розмір
                edgecolor='black',
                linewidth=2)

# Налаштування оформлення графіка
plt.title('Кластери методом Mean Shift', fontsize=16, fontweight='bold')
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Кластери')
plt.tight_layout()
plt.show()
