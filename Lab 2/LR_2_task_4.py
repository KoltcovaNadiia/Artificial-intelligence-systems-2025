import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt

# Зчитування даних   
input_file = 'income_data.txt'

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 5000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue

        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            y.append(0)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            y.append(1)
            count_class2 += 1

X = np.array(X)

#  Перетворення категоріальних даних у числові   
label_encoder = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoder.append(le)

X = X_encoded[:, :-1].astype(int)
y = np.array(y)

#  Розбиття на тренувальні та тестові дані   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

#   Порівняння моделей   
models = [
    ('LR', LogisticRegression(solver='liblinear')),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(kernel='rbf', gamma='scale'))  # покращений варіант
]

results = []
names = []
f1_scorer = make_scorer(f1_score, average='weighted')

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    f1_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=f1_scorer)
    results.append(cv_results)
    names.append(name)
    print(f"{name} - Accuracy: {cv_results.mean():.4f} ({cv_results.std():.4f})")
    print(f"{name} - F1 Score: {f1_results.mean():.4f} ({f1_results.std():.4f})\n")

# Побудова графіку точності   
plt.boxplot(results, tick_labels=names, patch_artist=True,
            boxprops=dict(facecolor='lavender'),
            medianprops=dict(color='darkblue'))
plt.title('Порівняння точності алгоритмів класифікації', fontsize=14, color='darkred')
plt.ylabel('Accuracy')
plt.xticks(rotation=20)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

#  Тренування найкращої моделі (SVM)   
model = SVC(kernel='rbf', gamma='scale')
model.fit(X_train, y_train)
predictions = model.predict(X_test)

#   Оцінка моделі   
print("\n   Оцінка моделі SVM   ")
print("Точність (Accuracy): {:.2f}%".format(accuracy_score(y_test, predictions) * 100))
print("\nМатриця помилок:")
print(confusion_matrix(y_test, predictions))
print("\nЗвіт про класифікацію:")
print(classification_report(y_test, predictions, zero_division=0))

#  Прогноз для нового зразка   
# Використаємо реальний приклад із тестових даних
X_new = X_test[0].reshape(1, -1)
prediction = model.predict(X_new)
predicted_class = '<=50K' if prediction[0] == 0 else '>50K'

print("\n   Новий прогноз   ")
print("Приклад із тестових даних:", X_new)
print("Імовірний дохід:", predicted_class)
