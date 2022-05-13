import numpy as np
import matplotlib.pyplot as plt
import pandas as ps

data = ps.read_csv('TSLA.csv')
x = data.iloc[:,[1,2,3,6]].values
y = data.iloc[:,4].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(x_train, y_train)
print(regressor.score(x_test, y_test))

y_pred = regressor.predict(x_test)


