import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load your dataset
data = pd.read_csv(r'C:\Users\CHARISHMA\Desktop\task 2\car data.csv')

# Display the first few rows of the dataset
print(data.head())

# Handle missing values
data = data.dropna()

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, drop_first=True)

# Define features and target variable
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Save the trained model to a file
joblib.dump(model, 'car_price_model.pkl')
