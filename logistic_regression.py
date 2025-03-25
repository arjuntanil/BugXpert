# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Step 2: Load the dataset
file_path = "LR_Dataset.csv"  # Ensure this file is in the working directory
df = pd.read_csv(file_path)

# Display dataset information
print("Dataset shape:", df.shape)
print("\nFirst few rows of the dataset:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())

# Step 3: Select independent (X) and dependent (y) variables
X = df.iloc[:, 1:5]  # Selecting 'Lines of Code', 'Complexity Score', 'Bug Frequency', 'Execution Time (ms)'
y = df.iloc[:, 5]    # Selecting 'Critical Bug (0=No, 1=Yes)'

# Check class distribution
print("\nClass distribution:")
print(y.value_counts())
print(f"Percentage of Critical Bugs: {y.mean()*100:.2f}%")

# Step 4: Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1 (Critical Bug)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print results
print("\n===== Model Evaluation =====")
print(f"Model Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Step 9: Feature Importance
feature_names = ['Lines of Code', 'Complexity Score', 'Bug Frequency', 'Execution Time (ms)']
coefficients = model.coef_[0]

print("\nFeature Importance (Coefficients):")
for feature, coef in zip(feature_names, coefficients):
    print(f"- {feature}: {coef:.4f}")

# Step 10: ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Step 11: Visualizations
# Create figure with two subplots
plt.figure(figsize=(15, 6))

# Plot 1: ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Plot 2: Feature Importance
plt.subplot(1, 2, 2)
feature_importance = pd.Series(coefficients, index=feature_names)
feature_importance.sort_values().plot(kind='barh', color='skyblue')
plt.xlabel('Coefficient Value')
plt.title('Feature Importance for Bug Criticality')
plt.grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# Step 12: Make Sample Predictions
print("\n===== Sample Predictions =====")
sample_bugs = [
    [1000, 8, 8, 50],   # Expected: Non-Critical
    [2500, 28, 50, 240]  # Expected: Critical
]

for i, features in enumerate(sample_bugs):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    bug_status = "Critical" if prediction == 1 else "Non-Critical"
    print(f"\nSample Bug #{i+1}:")
    print(f"Features: Lines of Code={features[0]}, Complexity={features[1]}, Bug Frequency={features[2]}, Execution Time={features[3]}ms")
    print(f"Prediction: {bug_status} Bug")
    print(f"Probability of being Critical: {probability:.4f} ({probability*100:.2f}%)") 