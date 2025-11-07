from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Завантаження даних
iris_dataset = load_iris()
df = pd.DataFrame(iris_dataset['data'][:5], columns=iris_dataset['feature_names'])

print("Ключі iris_dataset: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Назви відповідей: {}".format(iris_dataset['target_names']))
print("Назва ознак: \n{}".format(iris_dataset['feature_names']))
print("Тип масиву data: {}".format(type(iris_dataset['data'])))
print("Форма масиву data: {}".format(iris_dataset['data'].shape))
print("Значення ознак перших п’яти прикладів:\n", df)
print("Тип масиву target: {}".format(type(iris_dataset['target'])))
print("Відповіді:\n{}".format(iris_dataset['target']))

#  =
# КРОК 2. Дослідження даних
#  
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

print("\n----------------------")
print("Розмірність датасету:", dataset.shape)
print("----------------------")
print(dataset.head(20))
print("----------------------")
print("Статистичне зведення:")
print(dataset.describe())
print("----------------------")
print("Розподіл за класами:")
print(dataset.groupby('class').size())


# Діаграма розмаху 
dataset.plot(kind='box', subplots=True, layout=(2, 2),
             sharex=False, sharey=False,
             color='darkcyan', title='Розподіл ознак ірисів (Boxplot)')
plt.suptitle("Діаграма розмаху характеристик ірисів", fontsize=14, color='navy')
plt.show()

# Гістограми 
dataset.hist(color='orchid', edgecolor='black')
plt.suptitle("Гістограма розподілу атрибутів датасета", fontsize=14, color='darkmagenta')
plt.show()

# Матриця діаграм розсіювання 
scatter_matrix(dataset, figsize=(8, 8), diagonal='hist', color='teal')
plt.suptitle("Матриця діаграм розсіювання ознак ірисів", fontsize=14, color='darkgreen')
plt.show()


# Розділення датасету на навчальну та контрольну вибірки

array = dataset.values
X = array[:, 0:4]
y = array[:, 4]

X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1)

# Завантажуємо алгоритми моделі
models = []
models.append(('Логістична регресія', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('Лінійний дискримінантний аналіз', LinearDiscriminantAnalysis()))
models.append(('K-найближчих сусідів', KNeighborsClassifier()))
models.append(('Дерево рішень', DecisionTreeClassifier()))
models.append(('Наївний Байєс', GaussianNB()))
models.append(('Метод опорних векторів', SVC(gamma='auto')))

# Оцінювання моделей
results = []
names = []
print("\n=== Оцінка якості моделей ===")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: середня точність = {cv_results.mean():.4f}, стандартне відхилення = {cv_results.std():.4f}")

# Порівняння алгоритмів (оновлений стиль)
plt.boxplot(results, labels=names, patch_artist=True,
            boxprops=dict(facecolor='lavender'),
            medianprops=dict(color='darkblue'))
plt.title('Порівняння точності алгоритмів класифікації ірисів', fontsize=14, color='darkred')
plt.ylabel('Accuracy')
plt.xticks(rotation=20)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Тренування кращої моделі (SVM) на навчальних даних
model = SVC(gamma='auto')
model.fit(X_train, y_train)
predictions = model.predict(X_validation)

# Оцінка точності прогнозу
print("\n=== Оцінка моделі SVM ===")
print("Точність (Accuracy): {:.2f}%".format(accuracy_score(y_validation, predictions) * 100))
print("\nМатриця помилок:")
print(confusion_matrix(y_validation, predictions))
print("\nЗвіт про класифікацію:")
print(classification_report(y_validation, predictions))


# Прогноз для нового зразка ірису
X_new = np.array([[5.0, 2.9, 1.0, 0.2]])
prediction = model.predict(X_new)
predicted_class = iris_dataset['target_names'][list(iris_dataset['target_names']).index(prediction[0])] \
    if prediction[0] in iris_dataset['target_names'] else prediction[0]

print("\n=== Новий прогноз ===")
print("Вхідні дані:", X_new)
print("Передбачений клас:", prediction[0])
print("Імовірний сорт ірису:", predicted_class)
