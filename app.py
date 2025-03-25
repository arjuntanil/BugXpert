from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib
import os
from knn_quality_model import predict_code_quality, get_neighbors_info, load_knn_dataset, preprocess_data, train_knn_model, evaluate_model

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Polynomial Regression Model Class
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
        # Fix the deprecation warning by explicitly extracting the min and max values
        x_min = np.min(self.X).item()  # Extract scalar value
        x_max = np.max(self.X).item()  # Extract scalar value
        X_grid = np.arange(x_min, x_max, 1).reshape(-1, 1)
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

# Logistic Regression Model Class
class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.scaler = StandardScaler()
        self.accuracy = None
        self.conf_matrix = None
        self.class_report = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = ['Lines of Code', 'Complexity Score', 'Bug Frequency', 'Execution Time (ms)']
        
    def train_model(self, dataset_path=None):
        # Load the dataset directly using hardcoded path
        df = pd.read_csv("LR_Dataset.csv")
        
        # Select independent (X) and dependent (y) variables
        self.X = df.iloc[:, 1:5].values  # Features: lines of code, complexity, bug frequency, execution time
        self.y = df.iloc[:, 5].values    # Target: Critical Bug (0=No, 1=Yes)
        
        # Split the dataset into Training and Testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Feature scaling
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Evaluate the model
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.conf_matrix = confusion_matrix(self.y_test, y_pred)
        self.class_report = classification_report(self.y_test, y_pred, output_dict=True)
        
        # Save the model
        model_dir = os.path.dirname(os.path.abspath(__file__))
        joblib.dump(self.model, os.path.join(model_dir, 'bug_classification_model.pkl'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'feature_scaler.pkl'))
        
        return self.accuracy
    
    def predict(self, features):
        """
        Predict if a bug is critical or non-critical based on input features.
        
        Args:
            features: list or array with 4 elements [lines_of_code, complexity_score, bug_frequency, execution_time]
        
        Returns:
            prediction: 0 for Non-Critical Bug, 1 for Critical Bug
            probability: probability of being a Critical Bug
        """
        # Convert input to numpy array and reshape
        features = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0][1]  # Probability of being Critical
        
        return int(prediction), float(probability)
    
    def get_model_performance(self):
        """
        Generate performance metrics and visualizations for the logistic regression model.
        
        Returns:
            dict: Dictionary containing metrics and plots
        """
        # Compute ROC curve and AUC
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Create ROC curve plot
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save ROC plot to memory
        roc_buf = io.BytesIO()
        plt.savefig(roc_buf, format='png')
        roc_buf.seek(0)
        roc_plot_url = base64.b64encode(roc_buf.getvalue()).decode('utf-8')
        plt.close()
        
        # Create feature importance plot
        # For logistic regression, the coefficients can be interpreted as feature importance
        coef = self.model.coef_[0]
        feature_importance = pd.Series(coef, index=self.feature_names)
        
        plt.figure(figsize=(10, 6))
        feature_importance.sort_values().plot(kind='barh', color='skyblue')
        plt.xlabel('Coefficient Value')
        plt.title('Feature Importance for Bug Criticality')
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Save feature importance plot to memory
        feat_buf = io.BytesIO()
        plt.savefig(feat_buf, format='png')
        feat_buf.seek(0)
        feature_plot_url = base64.b64encode(feat_buf.getvalue()).decode('utf-8')
        plt.close()
        
        # Prepare performance metrics
        metrics = {
            'accuracy': round(self.accuracy, 4),
            'precision': round(self.class_report['1']['precision'], 4),
            'recall': round(self.class_report['1']['recall'], 4),
            'f1_score': round(self.class_report['1']['f1-score'], 4),
            'confusion_matrix': self.conf_matrix.tolist(),
            'roc_auc': round(roc_auc, 4),
            'roc_plot': roc_plot_url,
            'feature_importance_plot': feature_plot_url
        }
        
        return metrics

# Initialize and train all models
poly_model = PolynomialRegressionModel()
poly_model.train_model()

lr_model = LogisticRegressionModel()
lr_model.train_model()

# Initialize KNN model
try:
    # Check if model already exists
    if not os.path.exists("code_quality_knn_model.pkl"):
        print("Training KNN model for code quality classification...")
        dataset = load_knn_dataset()
        X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler = preprocess_data(dataset)
        knn_model = train_knn_model(X_train_scaled, y_train)
        print("KNN model training completed")
    else:
        print("KNN model already exists")
except Exception as e:
    print(f"Error initializing KNN model: {str(e)}")

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict-bugs')
def predict_bugs_page():
    return render_template('predict_bugs.html')

@app.route('/classify-bugs')
def classify_bugs_page():
    return render_template('classify_bugs.html')

@app.route('/code-quality')
def code_quality_page():
    return render_template('code_quality.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/predict_bugs', methods=['POST'])
def predict_bugs():
    try:
        lines_of_code = int(request.form['lines_of_code'])
        prediction = poly_model.predict(lines_of_code)
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/predict_criticality', methods=['POST'])
def predict_criticality():
    try:
        # Get features from form
        lines_of_code = int(request.form['lines_of_code'])
        complexity_score = int(request.form['complexity_score'])
        bug_frequency = int(request.form['bug_frequency'])
        execution_time = int(request.form['execution_time'])
        
        # Make prediction
        features = [lines_of_code, complexity_score, bug_frequency, execution_time]
        prediction, probability = lr_model.predict(features)
        
        # Return result
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': round(probability, 4),
            'is_critical': bool(prediction),
            'status': "Critical Bug" if prediction == 1 else "Non-Critical Bug"
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/predict_code_quality', methods=['POST'])
def predict_code_quality_api():
    try:
        # Get features from form
        lines_of_code = int(request.form['lines_of_code'])
        complexity_score = int(request.form['complexity_score'])
        execution_time = int(request.form['execution_time'])
        defect_density = float(request.form['defect_density'])
        
        # Make prediction
        features = [lines_of_code, complexity_score, execution_time, defect_density]
        quality_label, probabilities, quality_code = predict_code_quality(features)
        
        # Get nearest neighbors for context
        neighbors = get_neighbors_info(features, k=3)
        
        # Return result in a format expected by the frontend
        return jsonify({
            'success': True,
            'quality_class': quality_label,  # Changed quality_label to quality_class to match frontend
            'quality_code': int(quality_code),
            'probabilities': [round(float(p), 4) for p in probabilities],
            'neighbors': neighbors
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/model_performance/poly')
def poly_model_performance():
    performance = poly_model.get_model_performance()
    return jsonify(performance)

@app.route('/model_performance/lr')
def lr_model_performance():
    performance = lr_model.get_model_performance()
    return jsonify(performance)

@app.route('/model_performance/knn')
def knn_model_performance():
    try:
        # Load model
        knn = joblib.load("code_quality_knn_model.pkl")
        label_encoder = joblib.load("code_quality_label_encoder.pkl")
        
        # Get fresh dataset to evaluate
        dataset = load_knn_dataset()
        X_train_scaled, X_test_scaled, y_train, y_test, _, _ = preprocess_data(dataset)
        
        # Evaluate model
        performance = evaluate_model(knn, X_test_scaled, y_test, label_encoder)
        
        return jsonify(performance)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("Starting BugXpert - Software Bug Prediction & Classification System...")
    print("Access the application at http://127.0.0.1:5000/")
    
    # Print all defined routes for debugging
    print("\nDefined Routes:")
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint}: {rule.rule}")
    
    app.run(debug=True) 