import pandas as pd

# 1. Вхідні дані — набір Play Tennis
data = [
    {"Outlook": "Sunny",    "Humidity": "High",    "Wind": "Weak",   "Play": "No"},
    {"Outlook": "Sunny",    "Humidity": "High",    "Wind": "Strong", "Play": "No"},
    {"Outlook": "Overcast", "Humidity": "High",    "Wind": "Weak",   "Play": "Yes"},
    {"Outlook": "Rain",     "Humidity": "High",    "Wind": "Weak",   "Play": "Yes"},
    {"Outlook": "Rain",     "Humidity": "Normal",  "Wind": "Weak",   "Play": "Yes"},
    {"Outlook": "Rain",     "Humidity": "Normal",  "Wind": "Strong", "Play": "No"},
    {"Outlook": "Overcast", "Humidity": "Normal",  "Wind": "Strong", "Play": "Yes"},
    {"Outlook": "Sunny",    "Humidity": "High",    "Wind": "Weak",   "Play": "No"},
    {"Outlook": "Sunny",    "Humidity": "Normal",  "Wind": "Weak",   "Play": "Yes"},
    {"Outlook": "Rain",     "Humidity": "Normal",  "Wind": "Weak",   "Play": "Yes"},
    {"Outlook": "Sunny",    "Humidity": "Normal",  "Wind": "Strong", "Play": "Yes"},
    {"Outlook": "Overcast", "Humidity": "High",    "Wind": "Strong", "Play": "Yes"},
    {"Outlook": "Overcast", "Humidity": "Normal",  "Wind": "Weak",   "Play": "Yes"},
    {"Outlook": "Rain",     "Humidity": "High",    "Wind": "Strong", "Play": "No"}
]

df = pd.DataFrame(data)

# 2. Частотні таблиці
print("\n=== Частотні таблиці (Frequency Tables) ===\n")

for feature in ["Outlook", "Humidity", "Wind"]:
    print(f"--- {feature} ---")
    print(pd.crosstab(df[feature], df["Play"]), "\n")

# 3. Апріорні ймовірності класів
p_yes = (df["Play"] == "Yes").mean()
p_no = (df["Play"] == "No").mean()

print("=== Апріорні ймовірності ===")
print(f"P(Yes) = {p_yes:.3f}")
print(f"P(No)  = {p_no:.3f}\n")

# 4. Функція побудови таблиць правдоподібності
def likelihood_table(feature):
    freq = pd.crosstab(df[feature], df["Play"])
    p_given_yes = (freq["Yes"] / freq["Yes"].sum()).round(3)
    p_given_no = (freq["No"] / freq["No"].sum()).round(3)
    
    return pd.DataFrame({
        f"P({feature}=value | Yes)": p_given_yes,
        f"P({feature}=value | No)": p_given_no
    })

# Таблиці правдоподібності
print("=== Таблиці правдоподібності (Likelihood Tables) ===\n")

outlook_lh = likelihood_table("Outlook")
humidity_lh = likelihood_table("Humidity")
wind_lh = likelihood_table("Wind")

print("Outlook:\n", outlook_lh, "\n")
print("Humidity:\n", humidity_lh, "\n")
print("Wind:\n", wind_lh, "\n")

# 5. Умови, для яких треба визначити результат
X_outlook = "Sunny"
X_humidity = "High"
X_wind = "Weak"

print("=== Вхідні умови ===")
print(f"Outlook = {X_outlook}")
print(f"Humidity = {X_humidity}")
print(f"Wind = {X_wind}\n")

# 6. Вибір ймовірностей P(feature=value | class)
p1_yes = outlook_lh.loc[X_outlook, "P(Outlook=value | Yes)"]
p1_no  = outlook_lh.loc[X_outlook, "P(Outlook=value | No)"]

p2_yes = humidity_lh.loc[X_humidity, "P(Humidity=value | Yes)"]
p2_no  = humidity_lh.loc[X_humidity, "P(Humidity=value | No)"]

p3_yes = wind_lh.loc[X_wind, "P(Wind=value | Yes)"]
p3_no  = wind_lh.loc[X_wind, "P(Wind=value | No)"]

# 7. Байєсівське обчислення
unnorm_yes = p1_yes * p2_yes * p3_yes * p_yes
unnorm_no  = p1_no  * p2_no  * p3_no  * p_no

evidence = unnorm_yes + unnorm_no

P_yes_x = unnorm_yes / evidence
P_no_x = unnorm_no / evidence

print("=== Результат (Posterior Probabilities) ===")
print(f"P(Yes | x) = {P_yes_x:.4f}")
print(f"P(No  | x) = {P_no_x:.4f}\n")

# 8. Висновок
print("=== Висновок ===")
if P_yes_x > P_no_x:
    print("Матч ВІДБУДЕТЬСЯ (Yes)")
else:
    print("Матч НЕ відбудеться (No)")
