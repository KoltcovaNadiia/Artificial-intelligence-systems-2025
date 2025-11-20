import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib
matplotlib.use('Agg')          # щоб графік зберігався у файл, а не відкривав вікно
import matplotlib.pyplot as plt

# 1. Вхідний файл для твого варіанту
input_file = 'data_regr_4.txt'

# 2. Завантаження даних
data = np.loadtxt(input_file, delimiter=',')

# Останній стовпець – y, перший (один) стовпець – X
X, y = data[:, :-1], data[:, -1]

# 3. Розбивка на train/test (80% / 20%)
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# 4. Створення та навчання регресора
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# 5. Прогноз для тестових даних
y_test_pred = regressor.predict(X_test)

# 6. Графік
plt.scatter(X_test, y_test, label='Реальні значення')
plt.plot(X_test, y_test_pred, linewidth=4, label='Прогноз')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.savefig('regr_4_regression.png')
plt.close()
print("Графік збережено у файлі regr_4_regression.png")

# 7. Оцінка якості моделі
print("Linear regressor performance (variant 4):")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error  =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# 8. Збереження моделі
output_model_file = 'model_regr_4.pkl'
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)

# 9. Завантаження моделі та ПЕРЕДБАЧЕННЯ для нових значень
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)

# приклад нових значень X, для яких хочемо зробити прогноз
X_new = np.array([[-2.0], [0.0], [2.0], [4.0]])
y_new_pred = regressor_model.predict(X_new)

print("\nПередбачення для нових значень X:")
for x_val, y_pred in zip(X_new.flatten(), y_new_pred):
    print(f"x = {x_val:.2f} -> y ≈ {y_pred:.2f}")
