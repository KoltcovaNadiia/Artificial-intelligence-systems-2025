import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter

# -------------------------------
# Функція для візуалізації класифікатора
# -------------------------------
def visualize_classifier(classifier, X, y, title, save_file=None):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='Pastel1')
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', marker='x', s=80, label='Class 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='o', s=80, label='Class 1')
    plt.title(title, fontsize=14)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()

# -------------------------------
# Функція для друку зрозумілого звіту
# -------------------------------
def print_classification_report(y_true, y_pred, dataset_name):
    print(f"\n{'='*60}")
    print(f"Performance on {dataset_name}")
    print(f"{'='*60}\n")
    
    class_names = ['Class 0', 'Class 1']
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)
    
    counts = Counter(y_true)
    print(f"Class distribution in {dataset_name}: {dict(counts)}")
    
    accuracy = np.mean(y_true == y_pred)
    print(f"Overall Accuracy: {accuracy:.2f}")
    print(f"{'='*60}\n")

# -------------------------------
# Парсер аргументів
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Extra Trees classifier with imbalance handling')
    parser.add_argument('--balance', action='store_true', help='Враховувати дисбаланс класів')
    parser.add_argument('--saveplots', action='store_true', help='Зберігати графіки у файли')
    return parser.parse_args()

# -------------------------------
# Основна функція
# -------------------------------
def main():
    args = parse_args()

    # Завантаження даних
    data = np.loadtxt('data_imbalance.txt', delimiter=',')
    X, y = data[:, :-1], data[:, -1].astype(int)

    # Візуалізація вхідних даних
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y==0,0], X[y==0,1], c='red', marker='x', s=80, label='Class 0')
    plt.scatter(X[y==1,0], X[y==1,1], c='blue', marker='o', s=80, label='Class 1')
    plt.title('Вхідні дані (Data Imbalance)', fontsize=14)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    if args.saveplots:
        plt.savefig('input_data_imbalance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Розбиття на навчальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Параметри класифікатора
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 42}
    if args.balance:
        params['class_weight'] = 'balanced'

    clf = ExtraTreesClassifier(**params)
    clf.fit(X_train, y_train)

    # Візуалізація навчальних даних
    visualize_classifier(
        clf, X_train, y_train, 
        'Навчальні дані (Training Data)', 
        save_file='training_data.png' if args.saveplots else None
    )

    # Візуалізація тестових даних
    y_test_pred = clf.predict(X_test)
    visualize_classifier(
        clf, X_test, y_test, 
        'Тестові дані (Test Data)', 
        save_file='test_data.png' if args.saveplots else None
    )

    # Вивід звіту у консоль
    print_classification_report(y_train, clf.predict(X_train), "Training Data")
    print_classification_report(y_test, y_test_pred, "Test Data")

if __name__ == "__main__":
    main()
