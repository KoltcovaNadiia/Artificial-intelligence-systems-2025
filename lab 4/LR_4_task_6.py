import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Генерація випадкових даних (варіант 4)
m = 100
X = 6 * np.random.rand(m, 1) - 5  # X = 6 * random - 5
y = 0.7 * X ** 2 + X + 3 + np.random.randn(m, 1)  # y = 0.7 * X^2 + X + 3 + шум

# Функція для побудови кривих навчання
def plot_learning_curve(model, X, y, title):
    train_errors, val_errors = [], []
    for m in range(1, len(X)):
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=m, random_state=42)
        model.fit(X_train, y_train)
        y_train_predict = model.predict(X_train)
        y_val_predict = model.predict(X_val)
        
        train_errors.append(mean_squared_error(y_train, y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    
    plt.plot(np.arange(1, len(X)), train_errors, label='Помилка на навчальних даних')
    plt.plot(np.arange(1, len(X)), val_errors, label='Помилка на перевірочних даних')
    plt.title(title)
    plt.xlabel('Розмір навчального набору')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)

# Лінійна регресія
linear_model = LinearRegression()

# Поліноміальна регресія (2-й ступінь)
poly_2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly_2 = poly_2.fit_transform(X)
poly_model_2 = LinearRegression()

# Поліноміальна регресія (10-й ступінь)
poly_10 = PolynomialFeatures(degree=10, include_bias=False)
X_poly_10 = poly_10.fit_transform(X)
poly_model_10 = LinearRegression()

# Побудова кривих навчання для лінійної регресії
plt.figure(figsize=(10, 6))
plot_learning_curve(linear_model, X, y, "Криві навчання для лінійної регресії")
plt.show()

# Побудова кривих навчання для поліноміальної регресії (2-й ступінь)
plt.figure(figsize=(10, 6))
plot_learning_curve(poly_model_2, X_poly_2, y, "Криві навчання для поліноміальної регресії (ступінь 2)")
plt.show()

# Побудова кривих навчання для поліноміальної регресії (10-й ступінь)
plt.figure(figsize=(10, 6))
plot_learning_curve(poly_model_10, X_poly_10, y, "Криві навчання для поліноміальної регресії (ступінь 10)")
plt.show()
