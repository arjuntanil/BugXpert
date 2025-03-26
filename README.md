# BugXpert - Software Bug Prediction & Classification System

BugXpert is a web-based application that uses machine learning models to predict and classify software bugs based on various code metrics.

## Features

- **Bug Prediction**: Uses Polynomial Regression to predict the number of bugs based on lines of code
- **Bug Classification**: Uses Logistic Regression to classify bugs as critical or non-critical
- **Code Quality Classification**: Uses K-Nearest Neighbors (KNN) to classify code quality as High, Medium, or Low

## Technologies Used

- Flask web framework
- Scikit-learn for machine learning models
- Pandas and NumPy for data processing
- Matplotlib for data visualization
- Bootstrap for UI components

## Local Development

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`
4. Access the application at http://127.0.0.1:5000/

## Deployment

This application is configured for easy deployment to Render:

1. Fork or clone this repository to your GitHub account
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Render will automatically build and deploy the application

## Dataset Information

- `PR_Dataset.csv`: Contains data for polynomial regression model (lines of code vs. number of bugs)
- `LR_Dataset.csv`: Contains data for logistic regression model (code metrics vs. bug criticality)
- `KNN_Dataset.csv`: Contains data for KNN model (code metrics vs. code quality)

## License

MIT License 