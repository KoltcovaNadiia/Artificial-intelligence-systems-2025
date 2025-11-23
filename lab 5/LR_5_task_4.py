import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# --- Завантаження, перемішування та поділ даних ---
housing_data = fetch_california_housing()
X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

# Розбиття на навчальний та тестовий набори (20% для тестування)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7
)

# --- Побудова та навчання моделі AdaBoost ---
regressor = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=4),  # Базовий оцінювач - дерево глибиною 4
    n_estimators=400,                                # Кількість дерев
    random_state=7
)

regressor.fit(X_train, y_train)

# --- Оцінка ефективності регресора ---
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

print("\nADABOOST REGRESSOR")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

# --- Вилучення та візуалізація важливості ознак ---
feature_importances = regressor.feature_importances_
feature_names = np.array(housing_data.feature_names)  # Перетворюємо у масив для індексації

# Нормалізація важливості ознак
feature_importances = 100.0 * (feature_importances / max(feature_importances))

# Сортування значень для графіка
index_sorted = np.flipud(np.argsort(feature_importances))
pos = np.arange(index_sorted.shape[0]) + 0.5

# Побудова стовпчастої діаграми
plt.figure(figsize=(12, 6))
plt.bar(pos, feature_importances[index_sorted], align='center')
plt.xticks(pos, feature_names[index_sorted], rotation=45, ha='right')
plt.ylabel('Relative Importance')
plt.title("Оцінка важливості ознак з використанням регресора AdaBoost")
plt.tight_layout()
plt.show()
