import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Check if the file exists
file_path = r'C:\Users\CHARISHMA\Desktop\task 3\Advertising.csv'
print("File Exists:", os.path.exists(file_path))

# Load the dataset
data = pd.read_csv(file_path)

print(data.head())
print(data.describe())
print(data.info())
# Handle missing values
data = data.dropna()

# Encode categorical variables if any
data = pd.get_dummies(data, drop_first=True)

# Split the data into features and target variable
X = data.drop('Sales', axis=1)  # Replace 'Sales' with your target column name
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
