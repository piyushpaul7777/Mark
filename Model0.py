# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Data
data = {
    'Mwe': [370, 370, 370, 370, 1650, 1650, 1650, 1650, 2900, 2900, 2900, 2900, 3800, 3800, 3800, 3800, 370, 370, 370, 370, 1650, 1650, 1650, 1650, 2900, 2900, 2900, 2900, 3800, 3800, 3800, 3800, 2900, 3800, 370, 1650, 370, 370, 370, 370, 1650, 2900, 370, 370, 1650, 2900],
    'MWc': [230, 400, 2000, 4000, 230, 400, 2000, 4000, 230, 400, 2000, 4000, 230, 400, 2000, 4000, 230, 400, 2000, 4000, 230, 400, 2000, 4000, 230, 400, 2000, 4000, 230, 400, 2000, 4000, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 230, 230, 400, 400],
    'r': [0.75, 1.00, 1.25, 1.50, 1.00, 0.75, 1.50, 1.25, 1.25, 1.50, 0.75, 1.00, 1.50, 1.25, 1.00, 0.75, 1.50, 0.75, 1.00, 1.25, 1.25, 1.00, 0.75, 1.50, 1.00, 1.25, 1.50, 0.75, 0.75, 1.50, 1.25, 1.00, 1.00, 1.00, 1.00, 1.00, 1.25, 1.25, 1.50, 1.50, 1.25, 1.00, 1.00, 1.00, 1.25, 1.25],
    'Tcure': [90, 170, 210, 130, 130, 210, 170, 90, 170, 90, 130, 210, 210, 130, 90, 170, 130, 90, 170, 210, 90, 130, 210, 170, 210, 170, 90, 130, 170, 210, 130, 90, 210, 210, 210, 170, 210, 170, 210, 170, 210, 170, 210, 170, 210, 170],
    'Measured ad': [8.3, 28.8, 1.5, 0.00, 18.0, 14.6, 3.3, 2.0, 17.7, 5.8, 5.7, 4.4, 15.3, 10.4, 1.2, 4.0, 31.9, 2.8, 1.2, 0.6, 9.9, 18.9, 5.9, 1.4, 23.1, 24.6, 4.4, 2.0, 15.5, 28.9, 13.5, 0.0, 24.0, 21.2, 29.0, 22.4, 27.3, 27.8, 28.3, 23.1, 22.4, 24.6, 20.5, 24.6, 27.9, 23.5, 25.7]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['Mwe', 'MWc', 'r', 'Tcure']]
y = df['Measured ad']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Feature importance
feature_importance = model.feature_importances_
features = X.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=features)
plt.title('Feature Importance in Gradient Boosting Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=3)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Adhesive Strength')
plt.show()
