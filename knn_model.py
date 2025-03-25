# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import io
import base64

# KNN Model for Bug Solution Recommendations
def load_knn_datasets():
    """Load the bug classification dataset and bug solutions dataset"""
    # Load the KNN dataset for bug classification
    knn_data = pd.read_csv("KNN_Dataset.csv")
    
    # Load the solutions database
    solutions_data = pd.read_csv("bug_solutions.csv")
    
    return knn_data, solutions_data

def preprocess_data(knn_data):
    """Preprocess the dataset for KNN model"""
    # Extract features and target
    X = knn_data.iloc[:, 1:5]  # 'Lines of Code', 'Complexity Score', 'Bug Frequency', 'Execution Time'
    y = knn_data.iloc[:, 5]    # 'Critical Bug (0=No, 1=Yes)'
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for later use
    joblib.dump(scaler, "knn_scaler.pkl")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_knn_model(X_train_scaled, y_train, n_neighbors=3):
    """Train the KNN model"""
    # Initialize and train the KNN model
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train_scaled, y_train)
    
    # Save the model
    joblib.dump(knn_model, "knn_model.pkl")
    
    return knn_model

def evaluate_model(knn_model, X_test_scaled, y_test):
    """Evaluate the KNN model performance"""
    # Make predictions
    y_pred = knn_model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Visualize the confusion matrix
    cm_plot = plot_confusion_matrix(y_test, y_pred)
    
    # Visualize the decision boundary for the top 2 features
    db_plot = None
    if X_test_scaled.shape[1] >= 2:
        db_plot = plot_decision_boundary(knn_model, X_test_scaled[:, :2], y_test)
    
    return {
        'accuracy': round(accuracy, 4),
        'precision': round(report['weighted avg']['precision'], 4),
        'recall': round(report['weighted avg']['recall'], 4),
        'f1_score': round(report['weighted avg']['f1-score'], 4),
        'confusion_matrix_plot': cm_plot,
        'decision_boundary_plot': db_plot
    }

def plot_confusion_matrix(y_true, y_pred):
    """Plot the confusion matrix"""
    plt.figure(figsize=(6, 5))
    cm = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Encode the plot as base64
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return plot_base64

def plot_decision_boundary(model, X, y):
    """Plot the decision boundary for two features"""
    plt.figure(figsize=(8, 6))
    
    # Define the grid
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict on the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='Blues')
    
    # Plot the data points
    for label, marker, color in zip([0, 1], ['o', '^'], ['blue', 'red']):
        idx = y == label
        plt.scatter(X[idx, 0], X[idx, 1], c=color, marker=marker, alpha=0.7, edgecolor='k')
    
    plt.title('KNN Decision Boundary (Top 2 Features)')
    plt.xlabel('Scaled Feature 1 (Lines of Code)')
    plt.ylabel('Scaled Feature 2 (Complexity Score)')
    
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Encode the plot as base64
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return plot_base64

def find_bug_solutions(bug_features, bug_type, language, solutions_data, k=3):
    """Find similar bugs and recommend solutions"""
    # Load the trained KNN model and scaler
    if os.path.exists("knn_model.pkl") and os.path.exists("knn_scaler.pkl"):
        knn_model = joblib.load("knn_model.pkl")
        scaler = joblib.load("knn_scaler.pkl")
    else:
        # If model doesn't exist, train it
        knn_data, _ = load_knn_datasets()
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(knn_data)
        knn_model = train_knn_model(X_train_scaled, y_train)
    
    # Scale the input features
    scaled_features = scaler.transform([bug_features])
    
    # Find k nearest neighbors
    distances, indices = knn_model.kneighbors(scaled_features, n_neighbors=k)
    
    # Classify the bug as critical or non-critical
    is_critical = knn_model.predict(scaled_features)[0]
    
    # Filter solutions by bug_type and language if specified
    filtered_solutions = solutions_data
    if bug_type:
        filtered_solutions = filtered_solutions[filtered_solutions['bug_type'].str.contains(bug_type, case=False, na=False)]
    if language:
        filtered_solutions = filtered_solutions[filtered_solutions['language'].str.contains(language, case=False, na=False)]
    
    # If we have specific solutions, return them
    if not filtered_solutions.empty:
        # Sort by relevance (you might define a custom relevance function)
        # For now, just return the top matches
        recommended_solutions = filtered_solutions.head(min(3, len(filtered_solutions))).to_dict('records')
    else:
        # If no specific solutions, return general solutions based on criticality
        if is_critical:
            # Return solutions for critical bugs (higher indices in our dataset)
            recommended_indices = indices[0][distances[0].argsort()][:k]
            recommended_solutions = [
                {
                    'bug_type': 'Critical Bug',
                    'language': 'General',
                    'solution': 'Investigate high complexity areas. Focus on code with many dependencies.',
                    'doc_link': 'https://github.com/bug-fixing-best-practices'
                }
            ]
        else:
            # Return solutions for non-critical bugs (lower indices in our dataset)
            recommended_indices = indices[0][distances[0].argsort()][:k]
            recommended_solutions = [
                {
                    'bug_type': 'Non-Critical Bug',
                    'language': 'General',
                    'solution': 'Apply standard debugging techniques. Check edge cases and input validation.',
                    'doc_link': 'https://github.com/bug-fixing-best-practices'
                }
            ]
    
    return {
        'is_critical': bool(is_critical),
        'recommended_solutions': recommended_solutions,
        'distances': distances[0].tolist(),
        'neighbor_indices': indices[0].tolist()
    }

# If this script is run directly, train and evaluate the model
if __name__ == "__main__":
    print("Training KNN Model for Bug Solution Recommendations...")
    
    # Load datasets
    knn_data, solutions_data = load_knn_datasets()
    print(f"Loaded KNN dataset with {len(knn_data)} entries")
    print(f"Loaded solutions dataset with {len(solutions_data)} entries")
    
    # Preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(knn_data)
    print("Data preprocessing completed")
    
    # Train the model
    knn_model = train_knn_model(X_train_scaled, y_train)
    print("KNN model trained and saved")
    
    # Evaluate the model
    performance = evaluate_model(knn_model, X_test_scaled, y_test)
    print(f"KNN Model Accuracy: {performance['accuracy']}")
    print(f"KNN Model Precision: {performance['precision']}")
    print(f"KNN Model Recall: {performance['recall']}")
    print(f"KNN Model F1 Score: {performance['f1_score']}")
    
    # Test with sample bug features
    sample_features = [1500, 15, 20, 150]  # Lines of Code, Complexity Score, Bug Frequency, Execution Time
    recommendations = find_bug_solutions(sample_features, None, None, solutions_data)
    
    print("\nSample Bug Recommendation:")
    print(f"Bug Classification: {'Critical' if recommendations['is_critical'] else 'Non-Critical'}")
    print("\nRecommended Solutions:")
    for i, sol in enumerate(recommendations['recommended_solutions']):
        print(f"{i+1}. {sol['bug_type']} ({sol['language']}): {sol['solution']}")
        print(f"   Documentation: {sol['doc_link']}") 