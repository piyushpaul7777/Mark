import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('adhesive_properties.csv')

# Assign the features and target variables
X = data[['ratio', 'Tcure', 'Mwe', 'Mwc']]
y = data['target']  # replace 'target' with the actual target variable name

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

# Predicting on the test set
y_pred_dt = dt.predict(X_test)

# Calculating evaluation metrics
dt_mse = mean_squared_error(y_test, y_pred_dt)
dt_r2 = r2_score(y_test, y_pred_dt)

print(f"Decision Tree Regression MSE: {dt_mse}, R2 Score: {dt_r2}")