import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import joblib
import os

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