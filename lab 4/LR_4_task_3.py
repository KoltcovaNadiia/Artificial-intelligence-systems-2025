# Імпортуємо необхідні бібліотеки
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import pandas as pd

# === 1. Завантаження даних ===
# Вказуємо шлях до файлу з даними
input_file = 'data_multivar_regr.txt'

# Завантажуємо дані
data = pd.read_csv(input_file, delimiter=',')  # Для читання даних з комами

# Вивести перші кілька рядків для перевірки даних
print(data.head())

# Розділяємо дані на ознаки (X) та ціль (y)
X = data.iloc[:, :-1].values  # всі стовпці, окрім останнього (ознаки)
y = data.iloc[:, -1].values   # останній стовпець (ціль)

# === 2. Розбиття даних на навчальний та тестовий набори ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Перевірка розмірів тренувальних та тестових даних
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# === 3. Створення та навчання лінійного регресора ===
linear_regressor = linear_model.LinearRegression()

# Навчання моделі на тренувальних даних
linear_regressor.fit(X_train, y_train)

# Прогнозування на тестових даних
y_pred = linear_regressor.predict(X_test)

# === 4. Виведення результатів ===
print("=== Рівняння лінійної регресії ===")
terms = " + ".join([f"({coef:.4f})*x{i+1}" for i, coef in enumerate(linear_regressor.coef_)])
print(f"y = {terms} + ({linear_regressor.intercept_:.4f})")

# === 5. Метрики точності ===
mse = sm.mean_squared_error(y_test, y_pred)
mae = sm.mean_absolute_error(y_test, y_pred)
r2 = sm.r2_score(y_test, y_pred)

print(f"\nСередньоквадратична похибка (MSE): {mse:.4f}")
print(f"Середня абсолютна похибка (MAE): {mae:.4f}")
print(f"Коефіцієнт детермінації (R²): {r2:.4f}")

# === 6. Створення поліноміального регресора ===
polynomial = PolynomialFeatures(degree=10)

# Трансформуємо тренувальні дані для поліноміального регресора
X_train_transformed = polynomial.fit_transform(X_train)

# Створення лінійного регресора для поліноміальних даних
poly_linear_model = linear_model.LinearRegression()

# Навчання поліноміального регресора
poly_linear_model.fit(X_train_transformed, y_train)

# Прогнозування на тестових даних
X_test_transformed = polynomial.transform(X_test)
y_poly_pred = poly_linear_model.predict(X_test_transformed)

# Виведення метрик точності для поліноміального регресора
print("\nРезультати поліноміального регресора:")
mse_poly = sm.mean_squared_error(y_test, y_poly_pred)
mae_poly = sm.mean_absolute_error(y_test, y_poly_pred)
r2_poly = sm.r2_score(y_test, y_poly_pred)

print(f"Середньоквадратична похибка (MSE): {mse_poly:.4f}")
print(f"Середня абсолютна похибка (MAE): {mae_poly:.4f}")
print(f"Коефіцієнт детермінації (R²): {r2_poly:.4f}")

# === 7. Прогнозування для вибіркової точки даних ===
datapoint = [[7.75, 6.35, 5.56]]

# Трансформація вибіркової точки для поліноміального регресора
poly_datapoint = polynomial.fit_transform(datapoint)

# Прогнозування для вибіркової точки лінійним регресором
linear_prediction = linear_regressor.predict(datapoint)
print("\nПрогноз лінійної регресії для вибіркової точки:", linear_prediction)

# Прогнозування для вибіркової точки поліноміальним регресором
poly_prediction = poly_linear_model.predict(poly_datapoint)
print("Прогноз поліноміальної регресії для вибіркової точки:", poly_prediction)
