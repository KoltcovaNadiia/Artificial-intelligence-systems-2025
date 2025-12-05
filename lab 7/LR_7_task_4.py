import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler

# === Завантаження даних ===
data = np.loadtxt('data_clustering.txt', delimiter=',')
print("Перші 5 рядків даних:\n", data[:5])
print("Кількість рядків даних (точок):", data.shape[0])

# === Нормалізація даних ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# === Кластеризація методом AffinityPropagation ===
# Параметр preference можна підібрати для отримання 5 кластерів
ap = AffinityPropagation(damping=0.9, preference=-10, random_state=42)
ap.fit(X_scaled)
labels = ap.labels_
num_labels = len(np.unique(labels))
print("\nЗнайдено кластерів:", num_labels)

# === Центри кластерів ===
cluster_centers = np.array([X_scaled[labels == i].mean(axis=0) for i in range(num_labels)])
print("\nКоординати центрів кластерів:\n", cluster_centers)

# === Візуалізація кластерів ===
plt.figure(figsize=(10, 8))
colors = plt.cm.tab20(np.linspace(0, 1, num_labels))
markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', 'h', '8']

for k, col in zip(range(num_labels), colors):
    class_members = labels == k
    plt.scatter(X_scaled[class_members, 0], X_scaled[class_members, 1],
                marker=markers[k % len(markers)], s=80, c=[col], edgecolor='k', label=f'Кластер {k+1}')
    plt.scatter(cluster_centers[k, 0], cluster_centers[k, 1],
                marker='X', s=200, c=[col], edgecolor='k')
    plt.text(cluster_centers[k, 0]+0.05, cluster_centers[k, 1]+0.05, f'C{k+1}', fontsize=12, fontweight='bold')

plt.title("Affinity Propagation")
plt.xlabel("Нормалізована ознака 1")
plt.ylabel("Нормалізована ознака 2")
plt.grid(True)
plt.legend(loc='best', fontsize=10)
plt.show()
