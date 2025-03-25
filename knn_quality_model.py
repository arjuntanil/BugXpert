# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Set a random seed for reproducibility
np.random.seed(42)

# Step 1: Load the dataset and clean it
dataset_path = "KNN_Dataset.csv"
dataset = pd.read_csv(dataset_path)
print(f"Loaded raw dataset with {len(dataset)} entries")

# Clean the dataset - drop NaN values and empty rows
dataset = dataset.dropna()
print(f"Cleaned dataset has {len(dataset)} entries after removing NaN values")

# Step 2: Preprocess the data
# Separate features and target
X = dataset.iloc[:, 1:-1].values  # Features (LOC, Complexity, Execution Time, Defect Density)
y = dataset.iloc[:, -1].values    # Target (Code Quality)

# Encode categorical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Find the optimal K value
k_values = range(1, min(21, len(X_train)))  # Ensure K is not greater than number of samples
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='euclidean')
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plot the accuracy vs K value
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, 'o-', linewidth=2, markersize=8)
plt.grid(True)
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. K Value')
plt.xticks(k_values)
plt.savefig('app/static/images/knn_accuracy.png')

# Get the optimal K value
optimal_k = k_values[accuracies.index(max(accuracies))]
print(f"Optimal K value: {optimal_k}")

# Step 4: Train the KNN model with the optimal K
knn = KNeighborsClassifier(n_neighbors=optimal_k, weights='uniform', metric='euclidean')
knn.fit(X_train_scaled, y_train)

# Step 5: Evaluate the model
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

print(f"KNN Model Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
for class_name, metrics in class_report.items():
    if class_name in label_encoder.classes_:
        print(f"Class {class_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")

# Step 6: Save the model and preprocessors
joblib.dump(knn, "code_quality_knn_model.pkl")
joblib.dump(scaler, "code_quality_scaler.pkl")
joblib.dump(label_encoder, "code_quality_label_encoder.pkl")
print("Model and preprocessors saved to disk")

# Define safe prediction function with error handling
def predict_code_quality(features):
    try:
        # Convert input to numpy array and reshape
        features = np.array(features).reshape(1, -1)
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        quality_code = knn.predict(features_scaled)[0]
        
        # Get probability distribution
        probabilities = knn.predict_proba(features_scaled)[0]
        
        # Convert code to label
        quality_label = label_encoder.inverse_transform([quality_code])[0]
        
        return quality_label, probabilities, quality_code
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        # Return default values if prediction fails
        return "Medium", np.array([0.0, 1.0, 0.0]), 1

def get_neighbors_info(features, k=3):
    try:
        # Convert input to numpy array and reshape
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Find k nearest neighbors (ensure k is not greater than available samples)
        k = min(k, len(dataset))
        distances, indices = knn.kneighbors(features_scaled, n_neighbors=k)
        
        # Get information about nearest neighbors
        neighbors_info = []
        for i, idx in enumerate(indices[0]):
            try:
                module_id = dataset.iloc[idx, 0]  # Module ID
                loc = dataset.iloc[idx, 1]  # Lines of Code
                complexity = dataset.iloc[idx, 2]  # Complexity Score
                execution_time = dataset.iloc[idx, 3]  # Execution Time
                defect_density = dataset.iloc[idx, 4]  # Defect Density
                quality = dataset.iloc[idx, 5]  # Code Quality (original label)
                
                neighbors_info.append({
                    'module_id': int(module_id) if not pd.isna(module_id) else 0,
                    'loc': int(loc) if not pd.isna(loc) else 0,
                    'complexity': int(complexity) if not pd.isna(complexity) else 0,
                    'execution_time': int(execution_time) if not pd.isna(execution_time) else 0,
                    'defect_density': float(defect_density) if not pd.isna(defect_density) else 0.0,
                    'quality': quality if not pd.isna(quality) else "Unknown",
                    'distance': float(distances[0][i])
                })
            except Exception as e:
                print(f"Error processing neighbor {idx}: {str(e)}")
                # Add a placeholder neighbor if processing fails
                neighbors_info.append({
                    'module_id': 0,
                    'loc': 0,
                    'complexity': 0,
                    'execution_time': 0,
                    'defect_density': 0.0,
                    'quality': "Unknown",
                    'distance': 9999.0
                })
        
        return neighbors_info
    except Exception as e:
        print(f"Error getting neighbors: {str(e)}")
        # Return empty list if neighbor search fails
        return []

# These functions are kept for API compatibility
def load_knn_dataset():
    dataset = pd.read_csv(dataset_path)
    return dataset.dropna()  # Return cleaned dataset

def preprocess_data(dataset):
    try:
        # Clean dataset first
        dataset = dataset.dropna()
        
        # Separate features and target
        X = dataset.iloc[:, 1:-1].values
        y = dataset.iloc[:, -1].values
        
        # Encode categorical labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        # Return None values if preprocessing fails
        return None, None, None, None, None, None

def train_knn_model(X_train_scaled, y_train):
    try:
        # Choose a safe K value (not greater than samples)
        k = min(optimal_k, len(X_train_scaled) - 1) if len(X_train_scaled) > 1 else 1
        knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='euclidean')
        knn.fit(X_train_scaled, y_train)
        return knn
    except Exception as e:
        print(f"Error training model: {str(e)}")
        # Return a default model if training fails
        return KNeighborsClassifier(n_neighbors=1)

def evaluate_model(knn, X_test_scaled, y_test, label_encoder):
    try:
        y_pred = knn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
        
        return {
            'accuracy': float(accuracy),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }
    except Exception as e:
        print(f"Error evaluating model: {str(e)}")
        # Return placeholder metrics if evaluation fails
        return {
            'accuracy': 0.0,
            'confusion_matrix': [[0]],
            'classification_report': {'precision': 0, 'recall': 0, 'f1-score': 0}
        }

# Print completion message
print("KNN model for Code Quality Classification is ready!")

# If the script is run directly, print a test prediction
if __name__ == "__main__":
    print("\nTest prediction:")
    test_features = [1000, 20, 150, 5]  # LOC, Complexity, Execution Time, Defect Density
    quality_label, probabilities, quality_code = predict_code_quality(test_features)
    print(f"Quality Label: {quality_label}")
    print(f"Quality Code: {quality_code}")
    print(f"Probabilities: {probabilities}")
    
    print("\nNearest Neighbors:")
    neighbors = get_neighbors_info(test_features, k=3)
    for i, neighbor in enumerate(neighbors):
        print(f"Neighbor {i+1}: {neighbor['quality']} quality, distance={neighbor['distance']:.4f}") 