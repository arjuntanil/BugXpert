# Polynomial Regression for Bug Prediction

This component of BugXpert uses Polynomial Regression to predict the number of bugs in a software release based on code complexity (measured in lines of code).

## How It Works

1. **Data Collection**: The model uses historical data from `PR_Dataset.csv` containing:
   - `lines_of_code`: Independent variable representing code complexity
   - `num_bugs`: Dependent variable representing the number of bugs

2. **Polynomial Transformation**: The model transforms the linear input feature (lines of code) into polynomial features to capture the non-linear relationship between code complexity and bug frequency.

3. **Model Training**: A linear regression model is fitted on the polynomial features.

4. **Prediction**: When given a new value for lines of code, the model predicts the expected number of bugs.

## Usage

### Standalone Script
To run the polynomial regression model as a standalone script:

```bash
python polynomial_regression.py
```

This will:
- Load the dataset
- Train the model
- Display the R² score
- Visualize the relationship between lines of code and number of bugs
- Make a sample prediction

### Within Flask Application
The model is also integrated into the Flask application:

1. Access the app at `http://127.0.0.1:5000/`
2. Enter the number of lines of code 
3. Click "Predict" to see the estimated number of bugs

## Model Performance

The polynomial regression model achieves a high R² score (typically around 0.99), indicating excellent prediction accuracy.

## Implementation Details

The implementation follows these key steps:

```python
# Load dataset
df = pd.read_csv("PR_Dataset.csv")

# Select features
X = df.iloc[:, 0:1].values  # 'lines_of_code'
y = df.iloc[:, 1].values    # 'num_bugs'

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply polynomial transformation
poly = PolynomialFeatures(degree=3)
X_poly_train = poly.fit_transform(X_train)

# Train the model
model = LinearRegression()
model.fit(X_poly_train, y_train)

# Make predictions
X_poly_test = poly.transform(X_test)
y_pred = model.predict(X_poly_test)
```

The model uses a polynomial of degree 3, which effectively captures the relationship between lines of code and number of bugs. 