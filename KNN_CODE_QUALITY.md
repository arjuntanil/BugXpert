# Code Quality Classification with K-Nearest Neighbors

## Overview

This documentation describes the K-Nearest Neighbors (KNN) implementation for classifying software module code quality in the BugXpert application. The KNN model analyzes various metrics of a software module to classify its code quality as High, Medium, or Low.

## Dataset Features

The KNN model for code quality classification uses the following features:

1. **Lines of Code (LOC)**: The size of the software module in terms of lines of code
2. **Complexity Score**: Cyclomatic complexity metric of the module
3. **Execution Time (ms)**: Average execution time in milliseconds
4. **Defect Density**: Number of defects per 1000 lines of code

## Quality Classes

The model classifies modules into three quality categories:

| Quality Class | Description | Typical Defect Density |
|---------------|-------------|------------------------|
| High | Few or no defects; well-structured, maintainable code | < 2.5 defects per 1000 LOC |
| Medium | Moderate defect density; acceptable structure | 2.5 - 6 defects per 1000 LOC |
| Low | High defect density; poor structure, high complexity | > 6 defects per 1000 LOC |

## Model Implementation

### Preprocessing

1. The categorical labels (High, Medium, Low) are encoded using `LabelEncoder`
2. Features are standardized using `StandardScaler` to ensure all variables contribute equally
3. The dataset is split into training (80%) and testing (20%) sets

### K Selection

The model uses K=3 nearest neighbors by default, which was determined to be optimal through cross-validation and error rate analysis. An analysis of error rates for different K values is visualized in the web interface.

### Distance Metric

The model uses Euclidean distance to find the nearest neighbors:

```
distance = sqrt((x1 - y1)² + (x2 - y2)² + ... + (xn - yn)²)
```

Where x and y are two different feature vectors.

## Making Predictions

When a new software module is analyzed:

1. The module's features (LOC, complexity, execution time, defect density) are collected
2. Features are standardized using the same scaler as the training data
3. The model finds the K nearest neighbors based on Euclidean distance
4. The quality class is determined by majority vote of these K neighbors
5. The model also provides probability distribution across the three classes

## Performance Metrics

The model's performance is measured using:

1. **Accuracy**: Percentage of correctly classified modules
2. **Precision**: Ability to correctly identify each quality class
3. **Recall**: Ability to find all modules of a particular quality class
4. **Confusion Matrix**: Visualization of prediction errors

## Improving the Model

The performance of the KNN model can be improved by:

1. Collecting more training data 
2. Experimenting with different distance metrics
3. Feature engineering (adding more relevant metrics)
4. Trying different preprocessing techniques
5. Implementing feature weighting based on domain knowledge

## Usage in BugXpert

Users can use the Code Quality Classification feature to:

1. Assess the quality of their software modules
2. Identify modules that need quality improvement
3. Understand similar modules and their characteristics
4. Make informed decisions about code refactoring priorities

For any questions or issues, please refer to the BugXpert documentation or contact support. 