import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv(r'датасет.csv', sep='\t')
print(df.head())
print(df.columns)

df.columns = df.columns.str.strip()
print(df.dtypes)

if 'price' in df.columns and df['price'].dtype == 'object':
    df['price'] = df['price'].str.replace(',', '.')
    df['price'] = df['price'].astype(float)

plt.scatter(df.area, df.price, color='red')
plt.xlabel('Площадь (кв.м.)')
plt.ylabel('Стоимость (млн.руб)')
plt.title('Скаттерплот площади и стоимости квартир')
plt.show()

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

predicted_price_38 = reg.predict([[38]])
print(f'Стоимость квартиры площадью 38 м²: {predicted_price_38[0]} млн. руб.')

predicted_price_200 = reg.predict([[200]])
print(f'Стоимость квартиры площадью 200 м²: {predicted_price_200[0]} млн. руб.')

predicted_prices = reg.predict(df[['area']])
print(predicted_prices)

a = reg.coef_[0]
b = reg.intercept_

print(f'Коэффициент a: {a}')
print(f'Свободный член b: {b}')

print(f'Цена = {a} * площадь + {b}')

plt.scatter(df.area, df.price, color='red')
plt.xlabel('Площадь (кв.м.)')
plt.ylabel('Стоимость (млн.руб)')
plt.plot(df.area, reg.predict(df[['area']]), color='blue') 
plt.title('Линейная регрессия площади и стоимости квартир')
plt.show()

pred = pd.read_csv(r'prediction_price.csv', sep='\t')

pred.columns = pred.columns.str.strip()

if 'area' in pred.columns and pred['area'].dtype == 'object':
    pred['area'] = pred['area'].str.replace(',', '.').astype(float)

p = reg.predict(pred[['area']])
pred['predicted prices'] = p

print(pred)
pred.to_excel('new.xlsx', index=False)