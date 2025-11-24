import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# 1. Завантаження даних
# -------------------------------
url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
df = pd.read_csv(url)
print("Початкові дані:")
print(df.head(3))

# -------------------------------
# 2. Обробка дат та створення ознак
# -------------------------------
df['start_date'] = pd.to_datetime(df['start_date'])
df['end_date'] = pd.to_datetime(df['end_date'])
df['trip_duration'] = (df['end_date'] - df['start_date']).dt.total_seconds() / 3600
df['month'] = df['start_date'].dt.month
df['weekday'] = df['start_date'].dt.weekday

# Видаляємо непотрібні колонки
df = df.drop(columns=['insert_date', 'start_date', 'end_date'])

# Видаляємо рядки з пропущеними критичними значеннями
df = df.dropna(subset=['price', 'train_class', 'fare'])

# -------------------------------
# 3. Кодування категоріальних змінних
# -------------------------------
label_cols = ['origin', 'destination', 'train_type', 'train_class', 'fare']
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -------------------------------
# 4. Категоризація цін
# -------------------------------
def categorize_price(price):
    if price < 40:
        return 0  # cheap
    elif price < 80:
        return 1  # medium
    else:
        return 2  # expensive

df['price_category'] = df['price'].apply(categorize_price)

# -------------------------------
# 5. Підготовка ознак та цільової змінної
# -------------------------------
X = df.drop(columns=['price', 'price_category'])
y = df['price_category']

# -------------------------------
# 6. GaussianNB
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

print("\n=== GaussianNB: Матриця плутанини ===")
print(confusion_matrix(y_test, y_pred_gnb))
print("\n=== GaussianNB: Звіт класифікації ===")
print(classification_report(y_test, y_pred_gnb, target_names=['Cheap', 'Medium', 'Expensive']))

# -------------------------------
# 7. MultinomialNB з OneHotEncoder
# -------------------------------
numeric_cols = ['trip_duration', 'month', 'weekday']
cat_cols = ['origin', 'destination', 'train_type', 'train_class', 'fare']

ct = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False), numeric_cols + cat_cols)
])

pipe = Pipeline([
    ("prep", ct),
    ("clf", MultinomialNB())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipe.fit(X_train, y_train)
y_pred_mnb = pipe.predict(X_test)

print("\n=== MultinomialNB: Матриця плутанини ===")
print(confusion_matrix(y_test, y_pred_mnb))
print("\n=== MultinomialNB: Звіт класифікації ===")
print(classification_report(y_test, y_pred_mnb, target_names=['Cheap', 'Medium', 'Expensive'], zero_division=0))
