import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Діапазони змінних
universe_temp = np.arange(-30, 51, 1)  # Температура повітря [°C]
universe_rate = np.arange(-5, 6, 0.1)  # Швидкість зміни температури [°C/с]
universe_angle = np.arange(-90, 91, 1)  # Кут повороту регулятора кондиціонера

# Створення нечітких змінних
temperature = ctrl.Antecedent(universe_temp, 'temperature')
rate_of_change = ctrl.Antecedent(universe_rate, 'rate_of_change')
ac_control = ctrl.Consequent(universe_angle, 'ac_control', defuzzify_method='centroid')

# Функції належності для температури
temperature['very_hot'] = fuzz.trapmf(temperature.universe, [30, 40, 50, 50])  # Дуже гаряча
temperature['hot'] = fuzz.trimf(temperature.universe, [20, 30, 40])  # Тепла
temperature['normal'] = fuzz.trimf(temperature.universe, [15, 20, 25])  # Нормальна
temperature['cold'] = fuzz.trimf(temperature.universe, [5, 10, 15])  # Холодна
temperature['very_cold'] = fuzz.trapmf(temperature.universe, [-30, -30, 0, 5])  # Дуже холодна

# Функції належності для швидкості зміни температури
rate_of_change['negative'] = fuzz.trapmf(rate_of_change.universe, [-5, -5, -1, 0])  # Від'ємна швидкість
rate_of_change['zero'] = fuzz.trimf(rate_of_change.universe, [-0.5, 0, 0.5])  # Швидкість 0
rate_of_change['positive'] = fuzz.trapmf(rate_of_change.universe, [0, 1, 5, 5])  # Позитивна швидкість

# Функції належності для контролю кондиціонером
ac_control['large_left'] = fuzz.trimf(ac_control.universe, [-90, -90, -60])  # Великий кут вліво (холод)
ac_control['small_left'] = fuzz.trimf(ac_control.universe, [-60, -30, 0])  # Малий кут вліво (холод)
ac_control['neutral'] = fuzz.trimf(ac_control.universe, [-10, 0, 10])  # Нейтральний
ac_control['small_right'] = fuzz.trimf(ac_control.universe, [0, 30, 60])  # Малий кут вправо (тепло)
ac_control['large_right'] = fuzz.trimf(ac_control.universe, [60, 90, 90])  # Великий кут вправо (тепло)

# Створення правил для системи
rule1 = ctrl.Rule(temperature['very_hot'] & rate_of_change['positive'], ac_control['large_left'])  # Дуже гаряча і позитивна зміна -> великий кут вліво
rule2 = ctrl.Rule(temperature['very_hot'] & rate_of_change['negative'], ac_control['small_left'])  # Дуже гаряча і від'ємна зміна -> малий кут вліво
rule3 = ctrl.Rule(temperature['hot'] & rate_of_change['positive'], ac_control['large_left'])  # Тепла і позитивна зміна -> великий кут вліво
rule4 = ctrl.Rule(temperature['hot'] & rate_of_change['negative'], ac_control['neutral'])  # Тепла і від'ємна зміна -> вимкнути
rule5 = ctrl.Rule(temperature['very_cold'] & rate_of_change['negative'], ac_control['large_right'])  # Дуже холодна і від'ємна зміна -> великий кут вправо
rule6 = ctrl.Rule(temperature['very_cold'] & rate_of_change['positive'], ac_control['small_right'])  # Дуже холодна і позитивна зміна -> малий кут вправо
rule7 = ctrl.Rule(temperature['cold'] & rate_of_change['negative'], ac_control['large_left'])  # Холодна і від'ємна зміна -> великий кут вліво
rule8 = ctrl.Rule(temperature['cold'] & rate_of_change['positive'], ac_control['neutral'])  # Холодна і позитивна зміна -> вимкнути
rule9 = ctrl.Rule(temperature['very_hot'] & rate_of_change['zero'], ac_control['large_left'])  # Дуже гаряча і швидкість 0 -> великий кут вліво
rule10 = ctrl.Rule(temperature['hot'] & rate_of_change['zero'], ac_control['small_left'])  # Тепла і швидкість 0 -> малий кут вліво
rule11 = ctrl.Rule(temperature['very_cold'] & rate_of_change['zero'], ac_control['large_right'])  # Дуже холодна і швидкість 0 -> великий кут вправо
rule12 = ctrl.Rule(temperature['cold'] & rate_of_change['zero'], ac_control['small_right'])  # Холодна і швидкість 0 -> малий кут вправо
rule13 = ctrl.Rule(temperature['normal'] & rate_of_change['positive'], ac_control['small_left'])  # Нормальна і позитивна зміна -> малий кут вліво
rule14 = ctrl.Rule(temperature['normal'] & rate_of_change['negative'], ac_control['small_right'])  # Нормальна і від'ємна зміна -> малий кут вправо
rule15 = ctrl.Rule(temperature['normal'] & rate_of_change['zero'], ac_control['neutral'])  # Нормальна і швидкість 0 -> вимкнути

# Створення системи керування
ac_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])

# Симуляція
ac_simulation = ctrl.ControlSystemSimulation(ac_system)

# Визначимо декілька прикладів для симуляції
examples = [
    {"temperature": 35, "rate_of_change": 1},  # Дуже гаряча, позитивна зміна
    {"temperature": 25, "rate_of_change": -1},  # Тепла, від'ємна зміна
    {"temperature": -5, "rate_of_change": -2},  # Дуже холодна, від'ємна зміна
    {"temperature": 20, "rate_of_change": 0},   # Нормальна, швидкість 0
    {"temperature": 10, "rate_of_change": 2},   # Холодна, позитивна зміна
]

# Для кожного прикладу виконуємо симуляцію
for i, example in enumerate(examples, 1):
    ac_simulation.input['temperature'] = example["temperature"]
    ac_simulation.input['rate_of_change'] = example["rate_of_change"]
    ac_simulation.compute()
    
    print(f"\n--- Приклад {i}: Температура {example['temperature']}°C, Швидкість зміни {example['rate_of_change']}°C/с ---")
    print(f"Поворот кондиціонера: {ac_simulation.output['ac_control']:.2f} градусів")

# Візуалізація результатів
temp_range_plot = np.linspace(universe_temp.min(), universe_temp.max(), 30)
rate_range_plot = np.linspace(universe_rate.min(), universe_rate.max(), 30)
temp_grid, rate_grid = np.meshgrid(temp_range_plot, rate_range_plot)

ac_output = np.zeros_like(temp_grid)

for i in range(temp_grid.shape[0]):
    for j in range(temp_grid.shape[1]):
        ac_simulation.input['temperature'] = temp_grid[i, j]
        ac_simulation.input['rate_of_change'] = rate_grid[i, j]
        ac_simulation.compute()
        ac_output[i, j] = ac_simulation.output.get('ac_control', 0)  # Без помилки, якщо ключ відсутній

# Побудова графіку
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(temp_grid, rate_grid, ac_output, cmap='coolwarm')
ax.set_xlabel('Температура (°C)')
ax.set_ylabel('Швидкість зміни температури (°C/с)')
ax.set_zlabel('Кут регулятора кондиціонера (градуси)')
ax.set_title('Поверхня управління кондиціонером')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
