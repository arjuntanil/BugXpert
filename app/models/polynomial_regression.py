import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

class PolynomialRegressionModel:
    def __init__(self, degree=3):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree)
        self.model = LinearRegression()
        self.r2_score = None
        self.mse = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def train_model(self, dataset_path=None):
        # Load the dataset directly using hardcoded path
        df = pd.read_csv("PR_Dataset.csv")
        
        # Select independent (X) and dependent (y) variables
        self.X = df.iloc[:, 0:1].values  # 'lines_of_code'
        self.y = df.iloc[:, 1].values    # 'num_bugs'
        
        # Split the dataset into Training and Testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Apply Polynomial Regression
        X_poly_train = self.poly.fit_transform(self.X_train)
        
        # Train the model
        self.model.fit(X_poly_train, self.y_train)
        
        # Evaluate the model
        X_poly_test = self.poly.transform(self.X_test)
        y_pred = self.model.predict(X_poly_test)
        
        # Calculate metrics
        self.r2_score = r2_score(self.y_test, y_pred)
        self.mse = mean_squared_error(self.y_test, y_pred)
        
        # Save the model
        model_dir = os.path.dirname(os.path.abspath(__file__))
        joblib.dump(self.model, os.path.join(model_dir, 'bug_prediction_model.pkl'))
        joblib.dump(self.poly, os.path.join(model_dir, 'polynomial_features.pkl'))
        
        return self.r2_score
    
    def predict(self, lines_of_code):
        # Convert input to numpy array and reshape
        lines_of_code = np.array([lines_of_code]).reshape(-1, 1)
        
        # Transform input with polynomial features
        X_poly = self.poly.transform(lines_of_code)
        
        # Make prediction
        prediction = self.model.predict(X_poly)[0]
        
        return prediction
    
    def get_model_performance(self):
        # Generate visualization
        plt.figure(figsize=(10, 6))
        
        # Plot actual data points
        plt.scatter(self.X, self.y, color='red', label='Actual Data')
        
        # Generate smooth curve for prediction line
        X_grid = np.arange(min(self.X), max(self.X), 1).reshape(-1, 1)
        X_poly_grid = self.poly.transform(X_grid)
        y_grid_pred = self.model.predict(X_poly_grid)
        
        # Plot regression line
        plt.plot(X_grid, y_grid_pred, color='blue', label=f'Polynomial Regression (degree={self.degree})')
        
        # Set labels and title
        plt.xlabel("Lines of Code")
        plt.ylabel("Number of Bugs")
        plt.title("Polynomial Regression - Bug Prediction")
        plt.legend()
        
        # Save plot to memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # Return performance metrics and plot
        return {
            'r2_score': round(self.r2_score, 4),
            'mse': round(self.mse, 4),
            'plot': plot_url
        } 