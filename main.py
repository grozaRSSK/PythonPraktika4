import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Настройка стиля Seaborn
sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams['figure.figsize'] = (12, 8)
print("Анализ данных об экзопланетах с помощью Seaborn")
print("=" * 60)

# 1. Загружаем встроенный датасет Planets
planets = sns.load_dataset('planets')
print(f"Размер датасета: {planets.shape}")
print(f"\nПервые 5 строк данных:")
print(planets.head())
print(f"\nИнформация о данных:")
print(planets.info())
print(f"\nПропущенные значения:")
print(planets.isnull().sum())

# Создаём бинарный признак по аналогии с survived (0 - Нет, 1 - Да)
planets['detected_by_transit'] = (planets['method'] == 'Transit').astype(int)

# 2. Количество обнаруженных транзитным методом и остальными
print("\n" + "=" * 60)
print("1. ВИЗУАЛИЗАЦИЯ ОБНАРУЖЕНИЯ")
print("=" * 60)

plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
sns.countplot(data=planets, x='detected_by_transit', palette=['lightcoral', 'lightgreen'])
# sns.countplot() - Подсчет категориальных данных
plt.title('Общее количество обнаруженных и не обнаруженных транзитным методом')
plt.xlabel('Обнаружена транзитом (0 - Нет, 1 - Да)')
plt.ylabel('Количество')
for i, count in enumerate(planets['detected_by_transit'].value_counts().sort_index()):
    plt.text(i, count + 5, str(count), ha='center', va='bottom', fontweight='bold')

    # 3. Открытия по годам с разбивкой по методу
    plt.subplot(2, 3, 3)
    sns.countplot(data=planets, x='year', hue='detected_by_transit', palette='coolwarm')
    plt.title('Открытия по годам')
    plt.xlabel('Год')
    plt.ylabel('Количество')
    plt.legend(['Остальные', 'Транзит'])

# 4. Распределение массы обнаруженных и не обнаруженных транзитом
plt.subplot(2, 3, 4)
sns.histplot(data=planets, x='mass', hue='detected_by_transit', kde=True, palette='viridis', alpha=0.6)
# sns.histplot() - гистограммы распределения
plt.title('Распределение массы')
plt.xlabel('Масса (в массах Юпитера)')
plt.ylabel('Плотность')
plt.legend(['Остальные', 'Транзит'])

# 5. Корреляционная матрица
plt.figure(figsize=(10, 8))
numeric_data = planets.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Корреляционная матрица числовых признаков', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("3. РАСПРЕДЕЛЕНИЕ МАССЫ И РАССТОЯНИЯ")
print("=" * 60)

# 6. Создаём группы по годам открытия
planets['year_group'] = pd.cut(planets['year'],
                                bins=[1989, 2000, 2010, 2014, 2020],
                                labels=['1990-е', '2000-е', '2010-2013', '2014-2019'])

plt.figure(figsize=(12, 6))
sns.countplot(data=planets, x='year_group', hue='detected_by_transit', palette='Set2')
plt.title('Обнаружения по годам (группы)')
plt.xlabel('Год открытия')
plt.ylabel('Количество')
plt.legend(['Остальные', 'Транзит'])

# 7. Визуализация статистики
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# Средняя масса
sns.barplot(data=planets, x='detected_by_transit', y='mass', ci=None, ax=axes[0], palette='viridis')
axes[0].set_title('Средняя масса обнаруженных и не обнаруженных транзитом')
axes[0].set_xlabel('Обнаружена транзитом (0 - Нет, 1 - Да)')
axes[0].set_ylabel('Средняя масса (в массах Юпитера)')
# Средний период обращения
sns.barplot(data=planets, x='detected_by_transit', y='orbital_period', ci=None, ax=axes[1], palette='magma')
axes[1].set_title('Средний период обращения обнаруженных и не обнаруженных транзитом')
axes[1].set_xlabel('Обнаружена транзитом (0 - Нет, 1 - Да)')
axes[1].set_ylabel('Средний период (дни)')
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("4. ОСНОВНЫЕ ВЫВОДЫ")
print("=" * 60)
# Автоматический анализ результатов
total_planets = len(planets)
transit_rate = planets['detected_by_transit'].mean() * 100
other_rate = (1 - planets['detected_by_transit'].mean()) * 100
mean_mass_transit = planets[planets['detected_by_transit'] == 1]['mass'].mean() * 100
mean_mass_other = planets[planets['detected_by_transit'] == 0]['mass'].mean() * 100

print(f"Всего записей в датасете: {total_planets}")
print(f"Обнаружено транзитным методом: {transit_rate:.1f}%")
print(f"Обнаружено другими методами: {other_rate:.1f}%")
print(f"Средняя масса (транзит): {mean_mass_transit:.2f}")
print(f"Средняя масса (остальные): {mean_mass_other:.2f}")