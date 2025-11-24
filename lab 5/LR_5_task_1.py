import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Функція для візуалізації меж класифікатора та точок
def visualize_classifier(classifier, X, y, title, markers=None, colors=None, savepath=None):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)

    if markers is None:
        markers = ['s', 'o', '^']
    if colors is None:
        colors = ['red', 'blue', 'green']

    for i, marker in enumerate(markers):
        pts = X[y == i]
        plt.scatter(pts[:, 0], pts[:, 1], c=colors[i], label=f'Class {i}', marker=marker, edgecolor='black', s=80)

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    if savepath:
        plt.savefig(savepath)
    plt.show()


# ---------------------------
# Парсер аргументів
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Random Forest / Extra Trees classifier')
    parser.add_argument('--classifier', required=True, choices=['rf', 'erf'],
                        help='Type of classifier: rf = Random Forest, erf = Extra Trees')
    parser.add_argument('--saveplots', action='store_true', help='Save plots to files')
    return parser.parse_args()


# ---------------------------
# Основна функція
# ---------------------------
def main():
    args = parse_args()

    # Завантаження даних
    data = np.loadtxt('data_random_forests.txt', delimiter=',')
    X, y = data[:, :2], data[:, 2].astype(int)

    # Розбиття на навчальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Вибір класифікатора
    params = {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
    if args.classifier == 'rf':
        clf = RandomForestClassifier(**params)
        classifier_name = "Random Forest"
    else:
        clf = ExtraTreesClassifier(**params)
        classifier_name = "Extra Trees (Extremely Randomized Trees)"

    # Навчання
    clf.fit(X_train, y_train)

    # Візуалізація вхідних даних
    visualize_classifier(clf, X, y, f'Input Data - {classifier_name}', savepath='input_data.png' if args.saveplots else None)

    # Візуалізація меж класифікатора на навчальних даних
    visualize_classifier(clf, X_train, y_train, f'Training Data - {classifier_name}', savepath='training_boundary.png' if args.saveplots else None)

    # Прогнозування на тестових даних
    y_pred = clf.predict(X_test)

    # Візуалізація меж класифікатора на тестових даних
    visualize_classifier(clf, X_test, y_test, f'Test Data - {classifier_name}', savepath='test_boundary.png' if args.saveplots else None)

    # Звіт по класифікації
    class_names = ['Class-0', 'Class-1', 'Class-2']
    print(f"\n--- {classifier_name} Performance on Training Data ---")
    print(classification_report(y_train, clf.predict(X_train), target_names=class_names))
    print(f"\n--- {classifier_name} Performance on Test Data ---")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Рівні довірливості для тестових точок
    test_points = np.array([[5, 5], [3, 6], [6, 4], [7, 2], [4, 4], [5, 2]])
    probs = clf.predict_proba(test_points)
    print("\n--- Confidence Levels for Test Points ---")
    for idx, point in enumerate(test_points):
        predicted_class = np.argmax(probs[idx])
        print(f'Datapoint {point} -> Predicted Class: {predicted_class}, Probabilities: {probs[idx]}')

    # Візуалізація тестових точок
    visualize_classifier(clf, test_points, np.zeros(len(test_points)), f'Test Points - {classifier_name}', savepath='test_points.png' if args.saveplots else None)


# Запуск
if __name__ == "__main__":
    main()
