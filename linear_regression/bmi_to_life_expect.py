import pandas as pd
from matplotlib.pyplot import scatter, plot, show
from sklearn.linear_model.base import LinearRegression

bmi_life_data = pd.read_csv('bmi_to_life_expect.csv')
x_data = bmi_life_data[['BMI']]
y_data = bmi_life_data[['Life expectancy']]
# print x_data, y_data
bmi_life_model = LinearRegression()
bmi_life_model.fit(x_data, y_data)

laos_life_exp = bmi_life_model.predict([50.00])
print(laos_life_exp)
scatter(x_data, y_data)
plot(x_data, bmi_life_model.predict(x_data))
show()
