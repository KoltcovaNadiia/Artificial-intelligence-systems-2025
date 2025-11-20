# Імпортуємо необхідні бібліотеки
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# === 1. Завантаження даних ===
# Завантажуємо набір даних по діабету
diabetes = datasets.load_diabetes()
X = diabetes.data  # Ознаки
y = diabetes.target  # Цільова змінна

# Перевіряємо перші кілька рядків даних
print("Ознаки (X):", X[:5])
print("Ціль (y):", y[:5])

# === 2. Розбиття на навчальну та тестову вибірки ===
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

# Перевірка розмірів даних
print(f"Розмір тренувальних даних: {Xtrain.shape}")
print(f"Розмір тестових даних: {Xtest.shape}")

# === 3. Створення та навчання моделі лінійної регресії ===
regr = linear_model.LinearRegression()

# Навчання моделі на тренувальних даних
regr.fit(Xtrain, ytrain)

# Прогнозування на тестових даних
ypred = regr.predict(Xtest)

# === 4. Розрахунок показників якості ===
# Коефіцієнти регресії та вільний член
print("\nКоефіцієнти регресії:", regr.coef_)
print("Вільний член (intercept):", regr.intercept_)

# Розрахунок показників точності моделі
r2 = r2_score(ytest, ypred)
mae = mean_absolute_error(ytest, ypred)
mse = mean_squared_error(ytest, ypred)

print(f"\nКоефіцієнт детермінації (R²): {r2:.4f}")
print(f"Середня абсолютна похибка (MAE): {mae:.4f}")
print(f"Середньоквадратична похибка (MSE): {mse:.4f}")

# === 5. Побудова графіків ===
# Створення графіку залежності між спостережуваними значеннями і передбаченими
fig, ax = plt.subplots(figsize=(8, 6))

# Показуємо фактичні значення і передбачення
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0), alpha=0.7, color='blue', label='Передбачені значення')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4, label='Ідеальна лінія')

# Ось і легенда
ax.set_xlabel('Виміряно', fontsize=14)
ax.set_ylabel('Передбачено', fontsize=14)
ax.set_title('Лінійна регресія: Спостережувані vs Передбачені значення', fontsize=16)
ax.legend(loc='best')

# Відображаємо графік
plt.tight_layout()
plt.show()
