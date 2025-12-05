import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

#  Завантаження вхідних даних
X = np.loadtxt('data_clustering.txt', delimiter=',')

#  Візуалізація вхідних даних
plt.figure()
plt.scatter(X[:, 0], X[:, 1], edgecolors='black', facecolors='none', s=80)
plt.title('Вхідні дані')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

#  Налаштування моделі KMeans
num_clusters = 5
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)

#  Навчання моделі
kmeans.fit(X)


# Побудова сітки для візуалізації кордонів кластерів
step_size = 0.01

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size),
                             np.arange(y_min, y_max, step_size))

#  Прогноз кластерів на всій сітці
output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
output = output.reshape(x_vals.shape)

#  Відображення областей кластерів, даних та центрів
plt.figure()
plt.imshow(output, interpolation='nearest',
           extent=(x_vals.min(), x_vals.max(),
                   y_vals.min(), y_vals.max()),
           cmap=plt.cm.Paired,
           aspect='auto',
           origin='lower')

# Вхідні точки
plt.scatter(X[:, 0], X[:, 1], edgecolors='black',
            facecolors='none', s=80)

# Центри кластерів
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], marker='o',
            s=210, linewidths=3, color='black', zorder=12)

plt.title('Границі кластерів методом k-середніх')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

#  Оцінка якості кластеризації
inertia = kmeans.inertia_
silhouette = metrics.silhouette_score(X, kmeans.labels_)

print("Сумарна квадратична помилка (Inertia):", inertia)
print("Silhouette score:", silhouette)
