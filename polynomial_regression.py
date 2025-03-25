# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 2: Load the dataset
df = pd.read_csv("PR_Dataset.csv")

# Step 3: Select independent (X) and dependent (y) variables
X = df.iloc[:, 0:1].values  # Selecting 'lines_of_code' as independent variable
y = df.iloc[:, 1].values    # Selecting 'num_bugs' as dependent variable

# Step 4: Split the dataset into Training and Testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Apply Polynomial Regression
degree = 3  # You can change the degree for better accuracy
poly = PolynomialFeatures(degree=degree)
X_poly_train = poly.fit_transform(X_train)  # Transform X_train to polynomial features

# Step 6: Train the model
model = LinearRegression()
model.fit(X_poly_train, y_train)

# Step 7: Predict using the trained model
X_poly_test = poly.transform(X_test)  # Transform X_test to polynomial features
y_pred = model.predict(X_poly_test)

# Step 8: Check accuracy using R² score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Step 9: Visualizing the results
# Fix the deprecation warning by explicitly extracting the min and max values
x_min = np.min(X).item()  # Extract scalar value
x_max = np.max(X).item()  # Extract scalar value
X_grid = np.arange(x_min, x_max, 1).reshape(-1, 1)  # For smooth curve
X_poly_grid = poly.transform(X_grid)
y_grid_pred = model.predict(X_poly_grid)

plt.scatter(X, y, color='red', label='Actual Data')
plt.plot(X_grid, y_grid_pred, color='blue', label='Polynomial Regression Fit')
plt.xlabel("Lines of Code")
plt.ylabel("Number of Bugs")
plt.title("Polynomial Regression - Bug Prediction")
plt.legend()
plt.show()

# Make a sample prediction
sample_loc = 2000  # Example: 2000 lines of code
sample_loc_array = np.array([[sample_loc]])
sample_loc_poly = poly.transform(sample_loc_array)
predicted_bugs = model.predict(sample_loc_poly)[0]
print(f"For {sample_loc} lines of code, predicted number of bugs: {predicted_bugs:.2f}") 