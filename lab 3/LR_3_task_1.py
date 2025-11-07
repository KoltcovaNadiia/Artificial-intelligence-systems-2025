import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Діапазони змінних
universe_temp = np.arange(0, 101, 1)  # Температура води [0, 100] °C
universe_pressure = np.arange(0, 11, 0.1)  # Тиск [0, 10] одиниць
universe_angle = np.arange(-90, 91, 1)  # Кут повороту кранів від -90 до 90 градусів

# Створення нечітких змінних
temperature = ctrl.Antecedent(universe_temp, 'temperature')
pressure = ctrl.Antecedent(universe_pressure, 'pressure')
hot_tap = ctrl.Consequent(universe_angle, 'hot_tap', defuzzify_method='centroid')
cold_tap = ctrl.Consequent(universe_angle, 'cold_tap', defuzzify_method='centroid')

# Функції належності для температури води
temperature['cold'] = fuzz.trapmf(temperature.universe, [0, 0, 10, 25])
temperature['cool'] = fuzz.trimf(temperature.universe, [15, 30, 45])
temperature['warm'] = fuzz.trimf(temperature.universe, [40, 50, 60])
temperature['not_very_hot'] = fuzz.trimf(temperature.universe, [55, 70, 85])
temperature['hot'] = fuzz.trapmf(temperature.universe, [75, 90, 100, 100])

# Функції належності для тиску
pressure['weak'] = fuzz.trapmf(pressure.universe, [0, 0, 2, 4])
pressure['not_very_strong'] = fuzz.trimf(pressure.universe, [3, 5, 7])
pressure['strong'] = fuzz.trapmf(pressure.universe, [6, 8, 10, 10])

# Функції належності для кутів кранів
hot_tap['LL'] = fuzz.trimf(hot_tap.universe, [-90, -90, -60])  # Кран гарячої води сильно закритий
hot_tap['ML'] = fuzz.trimf(hot_tap.universe, [-75, -45, -15])  # Кран гарячої води середньо закритий
hot_tap['SL'] = fuzz.trimf(hot_tap.universe, [-30, -15, 0])   # Кран гарячої води слабо закритий
hot_tap['NC'] = fuzz.trimf(hot_tap.universe, [-10, 0, 10])    # Кран гарячої води нейтральний
hot_tap['SR'] = fuzz.trimf(hot_tap.universe, [0, 15, 30])     # Кран гарячої води слабо відкритий
hot_tap['MR'] = fuzz.trimf(hot_tap.universe, [15, 45, 75])    # Кран гарячої води середньо відкритий
hot_tap['LR'] = fuzz.trimf(hot_tap.universe, [60, 90, 90])    # Кран гарячої води сильно відкритий

cold_tap['LL'] = hot_tap['LL'].mf
cold_tap['ML'] = hot_tap['ML'].mf
cold_tap['SL'] = hot_tap['SL'].mf
cold_tap['NC'] = hot_tap['NC'].mf
cold_tap['SR'] = hot_tap['SR'].mf
cold_tap['MR'] = hot_tap['MR'].mf
cold_tap['LR'] = hot_tap['LR'].mf

# Створення правил для системи
rule1 = ctrl.Rule(temperature['hot'] & pressure['strong'],
                  (hot_tap['ML'], cold_tap['MR']))  # Гаряча вода і сильний напір
rule2 = ctrl.Rule(temperature['hot'] & pressure['not_very_strong'],
                  (hot_tap['NC'], cold_tap['MR']))  # Гаряча вода і не дуже сильний напір
rule3 = ctrl.Rule(temperature['not_very_hot'] & pressure['strong'],
                  (hot_tap['SL'], cold_tap['NC']))  # Не дуже гаряча вода і сильний напір
rule4 = ctrl.Rule(temperature['not_very_hot'] & pressure['weak'],
                  (hot_tap['SR'], cold_tap['SR']))  # Не дуже гаряча вода і слабкий напір
rule5 = ctrl.Rule(temperature['warm'] & pressure['not_very_strong'],
                  (hot_tap['NC'], cold_tap['NC']))  # Тепла вода і не дуже сильний напір
rule6 = ctrl.Rule(temperature['cool'] & pressure['strong'],
                  (hot_tap['MR'], cold_tap['ML']))  # Прохолодна вода і сильний напір
rule7 = ctrl.Rule(temperature['cool'] & pressure['not_very_strong'],
                  (hot_tap['MR'], cold_tap['SL']))  # Прохолодна вода і не дуже сильний напір
rule8 = ctrl.Rule(temperature['cold'] & pressure['weak'],
                  (hot_tap['LR'], cold_tap['NC']))  # Холодна вода і слабкий напір
rule9 = ctrl.Rule(temperature['cold'] & pressure['strong'],
                  (hot_tap['ML'], cold_tap['MR']))  # Холодна вода і сильний напір
rule10 = ctrl.Rule(temperature['warm'] & pressure['strong'],
                   (hot_tap['SL'], cold_tap['SL']))  # Тепла вода і сильний напір
rule11 = ctrl.Rule(temperature['warm'] & pressure['weak'],
                   (hot_tap['SR'], cold_tap['SR']))  # Тепла вода і слабкий напір

# Створення системи керування
tap_control_system = ctrl.ControlSystem(
    [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11]
)

# Симуляція
tap_simulation = ctrl.ControlSystemSimulation(tap_control_system)

# Визначимо приклади для симуляції
examples = [
    {"temperature": 90, "pressure": 9},  # Гаряча вода, сильний напір
    {"temperature": 20, "pressure": 4},  # Прохолодна вода, слабкий напір
    {"temperature": 55, "pressure": 7},  # Тепла вода, середній напір
    {"temperature": 10, "pressure": 2},  # Холодна вода, слабкий напір
    {"temperature": 45, "pressure": 8},  # Тепла вода, сильний напір
    {"temperature": 80, "pressure": 3},  # Гаряча вода, не дуже сильний напір
]

# Для кожного прикладу виконуємо симуляцію
for i, example in enumerate(examples, 1):
    tap_simulation.input['temperature'] = example["temperature"]
    tap_simulation.input['pressure'] = example["pressure"]
    tap_simulation.compute()
    
    print(f"\n--- Приклад {i}: Температура {example['temperature']}°C, Напір {example['pressure']} ---")
    print(f"Поворот гарячого крана: {tap_simulation.output['hot_tap']:.2f} градусів")
    print(f"Поворот холодного крана: {tap_simulation.output['cold_tap']:.2f} градусів")

# Візуалізація результатів (поверхні управління)
temp_range_plot = np.linspace(universe_temp.min(), universe_temp.max(), 30)
pres_range_plot = np.linspace(universe_pressure.min(), universe_pressure.max(), 30)
temp_grid, pres_grid = np.meshgrid(temp_range_plot, pres_range_plot)

hot_tap_output = np.zeros_like(temp_grid)
cold_tap_output = np.zeros_like(temp_grid)

for i in range(temp_grid.shape[0]):
    for j in range(temp_grid.shape[1]):
        tap_simulation.input['temperature'] = temp_grid[i, j]
        tap_simulation.input['pressure'] = pres_grid[i, j]
        tap_simulation.compute()
        hot_tap_output[i, j] = tap_simulation.output.get('hot_tap', 0)  # Без помилки, якщо ключ відсутній
        cold_tap_output[i, j] = tap_simulation.output.get('cold_tap', 0)  # Без помилки, якщо ключ відсутній

# Побудова графіків
fig1 = plt.figure(figsize=(10, 7))
ax1 = fig1.add_subplot(111, projection='3d')
surf1 = ax1.plot_surface(temp_grid, pres_grid, hot_tap_output, cmap='coolwarm')
ax1.set_xlabel('Температура (°C)')
ax1.set_ylabel('Напір (0-10)')
ax1.set_zlabel('Кут гарячого крана (градуси)')
ax1.set_title('Поверхня управління гарячим краном')
fig1.colorbar(surf1, shrink=0.5, aspect=5)

fig2 = plt.figure(figsize=(10, 7))
ax2 = fig2.add_subplot(111, projection='3d')
surf2 = ax2.plot_surface(temp_grid, pres_grid, cold_tap_output, cmap='coolwarm')
ax2.set_xlabel('Температура (°C)')
ax2.set_ylabel('Напір (0-10)')
ax2.set_zlabel('Кут холодного крана (градуси)')
ax2.set_title('Поверхня управління холодним краном')
fig2.colorbar(surf2, shrink=0.5, aspect=5)

plt.show()
