import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Генерація даних (варіант 4) ---
m = 100
X = 6 * np.random.rand(m, 1) - 5  # X = 6 * random - 5
y = 0.7 * X ** 2 + X + 3 + np.random.randn(m, 1)  # y = 0.7 * X^2 + X + 3 + шум

# --- 2. Лінійна регресія ---
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)
y_pred_lin = linear_regressor.predict(X)

# --- 3. Поліноміальна регресія (ступінь 2) ---
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y)
y_pred_poly = poly_regressor.predict(X_poly)

# --- 4. Обчислення метрик ---
mse_lin = mean_squared_error(y, y_pred_lin)
r2_lin = r2_score(y, y_pred_lin)
mse_poly = mean_squared_error(y, y_pred_poly)
r2_poly = r2_score(y, y_pred_poly)

# --- 5. Виведення коефіцієнтів ---
print("=== Лінійна регресія ===")
print(f"Рівняння: y = {linear_regressor.coef_[0][0]:.3f} * X + ({linear_regressor.intercept_[0]:.3f})")
print(f"MSE = {mse_lin:.3f}, R² = {r2_lin:.3f}\n")

print("=== Поліноміальна регресія (ступінь 2) ===")
a1, a2 = poly_regressor.coef_[0]
b = poly_regressor.intercept_[0]
print(f"Рівняння: y = {a2:.3f} * X² + ({a1:.3f}) * X + ({b:.3f})")
print(f"MSE = {mse_poly:.3f}, R² = {r2_poly:.3f}")

# --- 6. Побудова графіка ---
X_new = np.linspace(-5, 5, 200).reshape(-1, 1)
y_lin_new = linear_regressor.predict(X_new)
y_poly_new = poly_regressor.predict(poly.transform(X_new))

# Графік
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', s=20, alpha=0.6, label='Згенеровані дані')
plt.plot(X_new, y_lin_new, 'r--', label='Лінійна регресія')
plt.plot(X_new, y_poly_new, 'green', label='Поліноміальна регресія')
plt.title("Порівняння лінійної та поліноміальної регресій")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

