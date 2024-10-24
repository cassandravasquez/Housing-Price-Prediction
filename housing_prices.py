import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (make sure the file path is correct)
df = pd.read_csv('C:/Users/cassa/OneDrive/Documents/OpenCV/train.csv')

# Basic data exploration (optional, but useful)
print("Data Info:")
print(df.info())  # Check for missing values and data types

print("\nSummary statistics:")
print(df.describe())  # Get summary statistics

# Select the key features for prediction
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
X = df[features]
y = df['SalePrice']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model using Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nMean Absolute Error: {mae}")

# Display the coefficients of the model
print("\nModel Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef}")

# Optional: Visualizing correlations between features
plt.figure(figsize=(10, 8))
sns.heatmap(df[features + ['SalePrice']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()