import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_excel('датасет-1.xlsx')

print("Первые 5 строк датасета:")
print(df.head())
print("\nТипы данных:")
print(df.dtypes)

plt.scatter(df['area'], df['price'], color='red')
plt.title('Зависимость цены от площади квартиры')
plt.xlabel('Площадь (кв.м.)')
plt.ylabel('Цена (млн. руб.)')
plt.grid(True)
plt.show()

reg = LinearRegression()
reg.fit(df[['area']], df['price'])

print("\nКоэффициенты модели:")
print(f"Коэффициент (a): {reg.coef_[0]:.4f}")
print(f"Интерсепт (b): {reg.intercept_:.4f}")
print(f"Уравнение модели: price = {reg.coef_[0]:.4f} * area + {reg.intercept_:.4f}")

area_38 = reg.predict([[38]])
area_200 = reg.predict([[200]])
print(f"\nПредсказанная цена для 38 кв.м.: {area_38[0]:.4f} млн. руб.")
print(f"Предсказанная цена для 200 кв.м.: {area_200[0]:.4f} млн. руб.")

plt.scatter(df['area'], df['price'], color='red', label='Фактические данные')
plt.plot(df['area'], reg.predict(df[['area']]), label='Линия регрессии')
plt.title('Линейная регрессия: цена vs площадь')
plt.xlabel('Площадь (кв.м.)')
plt.ylabel('Цена (млн. руб.)')
plt.legend()
plt.grid(True)
plt.show()

pred = pd.read_excel('prediction_price.xlsx')
p = reg.predict(pred[['area']])

pred['predicted prices'] = p

print("\nПредсказанные цены для новых данных:")
print(pred)

pred.to_excel('predicted_prices.xlsx', index=False)
print("\nРезультаты сохранены в файл 'predicted_prices.xlsx'")
