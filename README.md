# BugXpert

BugXpert is a Software Bug Detection and Classification System that helps software developers detect and fix bugs efficiently before release.

## 🛠️ Machine Learning Techniques Used

### 1️⃣ Polynomial Regression – Predict Number of Bugs in Software Release
- **Goal**: Estimate the number of software bugs before release based on code complexity.
- **How it Works**:
  - Collects historical software project data (lines of code, complexity, past bugs).
  - Trains a Polynomial Regression model to predict expected bug count.
  - Outputs the estimated number of bugs in the next release.
- **Example**:
  - Input: Code complexity, project size, number of developers, past bug trends.
  - Output: Predicted number of bugs in the upcoming release.

## Project Structure

```
BugXpert/
│
├── app/
│   ├── models/
│   │   ├── __init__.py
│   │   └── polynomial_regression.py
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── main.js
│   └── templates/
│       ├── base.html
│       └── index.html
│
├── PR_Dataset.csv
├── app.py
├── requirements.txt
└── README.md
```

## Setup and Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/BugXpert.git
cd BugXpert
```

2. Create a virtual environment:
```
python -m venv venv
```

3. Activate the virtual environment:
   - On Windows:
   ```
   venv\Scripts\activate
   ```
   - On macOS and Linux:
   ```
   source venv/bin/activate
   ```

4. Install the required packages:
```
pip install -r requirements.txt
```

5. Run the application:
```
python app.py
```

6. Open your web browser and go to `http://127.0.0.1:5000/` to access BugXpert.

## Future Enhancements

- **K-Nearest Neighbors (KNN)** - For bug classification
- **Logistic Regression** - For bug severity prediction

## Dataset

The project uses `PR_Dataset.csv` for training the Polynomial Regression model, which contains:
- `lines_of_code`: Independent variable representing code complexity
- `num_bugs`: Dependent variable representing the number of bugs

## License

This project is licensed under the MIT License - see the LICENSE file for details. 