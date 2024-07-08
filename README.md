 Gurushika1CT4DA2639
### Step 1: Import Libraries and Load Data

First, import the necessary libraries and load your dataset. For this example, let's use the Boston Housing dataset from scikit-learn.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Boston Housing dataset
boston = load_boston()
df = pd.DataFrame(data=boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target  # target variable (median house value)
```

### Step 2: Explore the Data

Let's take a quick look at the dataset to understand its structure and any potential correlations.

```python
# Display first few rows of the dataset
print(df.head())

# Check the shape of the dataset
print(df.shape)

# Get summary statistics
print(df.describe())
```

### Step 3: Prepare Data for Modeling

Define the feature (`X`) and target (`y`) variables. Here, we'll use 'RM' (average number of rooms per dwelling) as the feature.

```python
# Define X (feature) and y (target)
X = df[['RM']]  # using 'RM' as the feature
y = df['MEDV']
```

### Step 4: Split Data into Training and Testing Sets

Split the data into training and testing sets. We'll use 80% of the data for training and 20% for testing.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Step 5: Train the Linear Regression Model

Instantiate the LinearRegression model, fit it on the training data, and make predictions on the test data.

```python
# Instantiate the model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
```

### Step 6: Evaluate Model Performance

Evaluate the model using metrics like Mean Squared Error (MSE) and R-squared.

```python
# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')
```

### Step 7: Visualize the Regression Line and Predictions

Visualize the regression line along with the actual vs. predicted values.

```python
# Plot outputs
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title('Actual vs. Predicted Values')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('Median House Value (MEDV)')
plt.legend()
plt.grid(True)
plt.show()
```

### Conclusion

This example demonstrates a simple linear regression model using the Boston Housing dataset. After training the model, we evaluated its performance using Mean Squared Error (MSE) and R-squared metrics. Finally, we visualized the regression line along with the actual vs. predicted values to assess the model's accuracy. Adjust and customize these steps based on your specific dataset and analysis goals.
