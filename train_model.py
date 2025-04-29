import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Load the dataset
file_path = '/mnt/data/project data sets.csv'
data = pd.read_csv(file_path, encoding='latin1')

# Drop unnecessary columns
data = data.drop(['customer name', 'customer e-mail'], axis=1)

# Encode 'country' (categorical feature)
le = LabelEncoder()
data['country'] = le.fit_transform(data['country'])

# Feature matrix and target vector
X = data.drop('car purchase amount', axis=1)
y = data['car purchase amount']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Regressor
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

# Set up hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# Grid Search with Cross-Validation
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid,
                           cv=3, scoring='r2', verbose=1, n_jobs=-1)

grid_search.fit(X_train, y_train)

# Best model
best_xgb = grid_search.best_estimator_

# Make predictions
y_pred = best_xgb.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nModel Performance:")
print(f"R^2 Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Display best parameters
print("\nBest Hyperparameters:")
print(grid_search.best_params_)
