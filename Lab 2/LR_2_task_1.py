import numpy as np
from sklearn import preprocessing
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score

# Вхідний файл
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

def is_int_like(s: str) -> bool:
    s = s.strip()
    if s.startswith('-'):
        s = s[1:]
    return s.isdigit()

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue

        data = line.strip().split(', ')
        if not data:
            continue

        # останній елемент — мітка класу; 14 попередніх — ознаки
        label = data[-1]
        feats = data[:-1]

        if label == '<=50K' and count_class1 < max_datapoints:
            X.append(feats)
            y.append(label)
            count_class1 += 1
        elif label == '>50K' and count_class2 < max_datapoints:
            X.append(feats)
            y.append(label)
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X, dtype=object)
y = np.array(y, dtype=object)

# Перетворення рядкових даних на числові
label_encoder = {}            # зберігаємо енкодер для кожної категоріальної колонки за її індексом
X_encoded = np.zeros_like(X, dtype=int)

# визначаємо числові/категоріальні колонки по першому рядку
num_cols = []
cat_cols = []
for i, item in enumerate(X[0]):
    if is_int_like(str(item)):
        num_cols.append(i)
    else:
        cat_cols.append(i)

# числові — просто приводимо до int
for i in num_cols:
    X_encoded[:, i] = X[:, i].astype(int)

# категоріальні — кодуємо окремим LabelEncoder для кожної колонки
for i in cat_cols:
    le = preprocessing.LabelEncoder()
    X_encoded[:, i] = le.fit_transform(X[:, i])
    label_encoder[i] = le

# кодуємо ціль окремим енкодером
y_le = preprocessing.LabelEncoder()
y_enc = y_le.fit_transform(y)

X = X_encoded.astype(int)
y = y_enc.astype(int)

# Розбиття на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)

# Створення та навчання SVM-класифікатора
classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=10000))
classifier.fit(X_train, y_train)

# Передбачення
y_test_pred = classifier.predict(X_test)

# Обчислення F1-міри
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
accuracy = cross_val_score(classifier, X, y, scoring='accuracy', cv=3)
precision = cross_val_score(classifier, X, y, scoring='precision_weighted', cv=3)
recall = cross_val_score(classifier, X, y, scoring='recall_weighted', cv=3)

print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")
print("Accuracy: " + str(round(100 * accuracy.mean(), 2)) + "%")
print("Precision: " + str(round(100 * precision.mean(), 2)) + "%")
print("Recall: " + str(round(100 * recall.mean(), 2)) + "%")

# Тестова точка
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
              '0', '0', '40', 'United-States']

# Кодування тестової точки
input_data_encoded = []
for i, item in enumerate(input_data):
    if i in num_cols:
        input_data_encoded.append(int(item))
    else:
        le = label_encoder[i]
        try:
            input_data_encoded.append(int(le.transform([item])[0]))
        except ValueError:
            # якщо нова (небачена) категорія — обережний фолбек
            input_data_encoded.append(0)

input_data_encoded = np.array(input_data_encoded, dtype=int)

# Передбачення класу
predicted_class = classifier.predict([input_data_encoded])
predicted_label = y_le.inverse_transform(predicted_class)[0]
print(predicted_label)

# Висновок
print("Висновок: щорічний дохід цієї особи, за моделлю, "
      + ("перевищує $50K." if predicted_label == ">50K" else "не перевищує $50K."))
