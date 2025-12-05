import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
X = iris['data']
y = iris['target']

kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel('Довжина чашолистка (cm)')
plt.ylabel('Ширина чашолистка (cm)')
plt.title('Кластеризація Iris методом K-середніх')
plt.show()

print("Координати центрів кластерів:\n", kmeans.cluster_centers_)
print("\nКластерні мітки для перших 10 зразків:\n", y_kmeans[:10])
