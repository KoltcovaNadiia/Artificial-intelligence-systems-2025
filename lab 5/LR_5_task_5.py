import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split

# --- 1. Завантаження та попередня обробка даних ---

input_file = 'traffic_data.txt'
data = []

with open(input_file, 'r') as f:
    for line in f.readlines():
        items = line.strip().split(',')
        data.append(items)

data = np.array(data)

# --- 2. Кодування категоріальних ознак ---
label_encoders = []
X_encoded = np.empty(data.shape, dtype=int)

for i in range(data.shape[1]):
    # Перевіряємо, чи числовий стовпець
    try:
        X_encoded[:, i] = data[:, i].astype(int)
    except ValueError:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(data[:, i])
        label_encoders.append(le)

# Відокремлюємо ознаки та цільову змінну
X = X_encoded[:, :-1]  # ознаки
y = X_encoded[:, -1]   # кількість транспортних засобів

# --- 3. Розбиття на навчальний та тестовий набори ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# --- 4. Створення та навчання класифікатора Extra Trees ---
clf = ExtraTreesClassifier(
    n_estimators=200,    # кількість дерев
    max_depth=15,        # максимальна глибина дерев
    random_state=42
)
clf.fit(X_train, y_train)

# --- 5. Оцінка ефективності моделі ---
y_pred = clf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Accuracy on test set: {accuracy:.2f}")

# --- 6. Прогноз для нової точки даних ---
test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
test_encoded = []

le_index = 0
for i, item in enumerate(test_datapoint):
    try:
        test_encoded.append(int(item))
    except ValueError:
        test_encoded.append(int(label_encoders[le_index].transform([item])[0]))
        le_index += 1

test_encoded = np.array(test_encoded).reshape(1, -1)
predicted = clf.predict(test_encoded)[0]
print(f"Predicted traffic intensity: {predicted}")
