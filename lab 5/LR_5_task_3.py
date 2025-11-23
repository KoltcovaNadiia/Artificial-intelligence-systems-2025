import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

# Візуалізація вхідних даних
def visualize_input_data(X, y, title, save_file=None):
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green']
    markers = ['x', 'o', 's']
    for i in range(3):
        plt.scatter(X[y == i, 0], X[y == i, 1], c=colors[i], marker=markers[i], s=80, label=f'Class {i}')
    plt.title(title, fontsize=14)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()

# Візуалізація зон класифікації
def visualize_classifier(classifier, X, y, title, save_file=None):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='Pastel1')
    colors = ['red', 'blue', 'green']
    markers = ['x', 'o', 's']
    for i in range(3):
        plt.scatter(X[y == i, 0], X[y == i, 1], c=colors[i], marker=markers[i], s=80, label=f'Class {i}')
    plt.title(title, fontsize=14)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()

# Основна функція
def main():
    # Завантаження даних
    input_file = 'data_random_forests.txt'
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1].astype(int)  # Приводимо мітки до int

    # Візуалізація вхідних даних
    visualize_input_data(X, y, 'Вхідні дані (Random Forest)', save_file='input_data.png')

    # Розбиття на навчальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Визначення сітки параметрів для Grid Search
    parameter_grid = [
        {'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]},
        {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]}
    ]

    metrics = ['precision_weighted', 'recall_weighted']

    # Сітковий пошук для кожної метрики
    for metric in metrics:
        print(f"\n### Searching optimal parameters for {metric} ###")
        clf = GridSearchCV(
            ExtraTreesClassifier(random_state=42),
            parameter_grid,
            cv=5,
            scoring=metric
        )
        clf.fit(X_train, y_train)

        # Вивід результатів Grid Search
        print("\nGrid scores for the parameter grid:")
        for mean_score, params in zip(clf.cv_results_['mean_test_score'], clf.cv_results_['params']):
            print(f"{params} --> {mean_score:.3f}")

        print("\nBest parameters:", clf.best_params_)

        # Вивід звіту на тестових даних
        y_pred = clf.predict(X_test)
        print(f"\nPerformance report for metric: {metric}\n")
        print(classification_report(y_test, y_pred))

        # Візуалізація класифікації на тестових даних
        visualize_classifier(
            clf.best_estimator_,
            X_test,
            y_test,
            f'Test Data Classification ({metric})',
            save_file=f'test_classification_{metric}.png'
        )

if __name__ == "__main__":
    main()
