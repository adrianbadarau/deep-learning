import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

data = pd.read_fwf('brain_body.txt')
x_values = data[['Brain']]
y_values = data[['Body']]
print(x_values, y_values)

# train model
model = linear_model.LinearRegression()
model.fit(x_values, y_values)

# visualize data
plt.scatter(x_values, y_values)
plt.plot(x_values, model.predict(x_values))
plt.show()
