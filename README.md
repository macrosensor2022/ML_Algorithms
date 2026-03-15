# ML Algorithms

A collection of machine learning algorithm implementations using Python and scikit-learn.

## 📚 Contents

This repository contains implementations of fundamental machine learning algorithms with detailed explanations and visualizations:

1. **Linear Regression** - Predicting continuous values
2. **Logistic Regression (Binary)** - Student pass/fail prediction
3. **Logistic Regression (Insurance)** - Bought-insurance prediction
4. **Multiclass Logistic Regression (HR Analytics)** - Employee attrition prediction
5. **Train/Test Split & Evaluation** - Practical workflow with accuracy metrics
6. **Multivariate Linear Regression** - Predicting values using multiple features
7. **Gradient Descent** - Custom implementation of optimization algorithm
8. **Decision Tree (Salary)** - Predicting salary brackets using categorical features
9. **Decision Tree (Titanic Survival)** - Survival prediction with tree-based model
10. **Support Vector Machine (SVM)** - Iris flower classification with kernel trick
11. **Random Forest** - Digit recognition and Titanic survival with ensemble learning

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.x installed along with the following libraries:

```bash
pip install numpy pandas matplotlib scikit-learn word2number
```

### Running the Notebooks

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/ML-algorithms.git
cd ML-algorithms
```

2. Open Jupyter Notebook:
```bash
jupyter notebook
```

3. Navigate to the desired notebook and run the cells

## 📊 Algorithms

### 1. Linear Regression

**File:** `Linear_regression.ipynb`

**Description:** Demonstrates linear regression for predicting student scores based on study hours.

**Key Features:**
- Dataset creation and visualization
- Model training and parameter extraction
- Performance metrics (R², MSE, RMSE)
- Prediction visualization with regression line

**Use Case:** Predicting continuous numerical values (e.g., prices, scores, temperatures)

---

### 2. Logistic Regression (Binary)

**File:** `Logistic_regression.ipynb`

**Description:** Implements binary classification to predict student pass/fail outcomes based on study hours.

**Key Features:**
- Binary classification (Pass/Fail)
- Probability estimation
- Sigmoid curve visualization
- Decision boundary analysis
- Model parameter interpretation

**Use Case:** Binary classification problems (e.g., spam detection, disease diagnosis, pass/fail prediction)

---

### 3. Logistic Regression (Insurance Purchase)

**File:** `log_regression.ipynb`

**Description:** Logistic regression model to predict whether a person buys insurance based on age, including a clear explanation of the logistic function and decision boundary.

**Key Features:**
- Binary classification (Buys Insurance / Does Not Buy)
- Visualization of age vs purchase decision
- Model training using scikit-learn
- Probability prediction for new ages
- Interpretation of model coefficients and decision threshold

**Use Case:** Simple binary decision-making based on a single numeric feature.

---

### 4. Multiclass Logistic Regression (HR Analytics)

**File:** `logistic_regression_multiclass.ipynb`

**Description:** Multiclass logistic regression on an HR dataset to predict whether an employee will leave the company, with extensive feature engineering and evaluation.

**Key Features:**
- Multiclass / multinomial logistic regression
- One-hot encoding for categorical features
- Train/test split and accuracy evaluation
- Confusion matrix and classification report
- Detailed comments explaining each preprocessing and modeling step

**Use Case:** Employee attrition prediction and similar HR analytics classification tasks.

---

### 5. Train/Test Split & Model Evaluation

**File:** `train_test.ipynb`

**Description:** Demonstrates how to correctly split data into training and test sets, train a multivariate linear regression model on car prices, and evaluate performance.

**Key Features:**
- `train_test_split` usage with `test_size` and `random_state` (with comments explaining each parameter)
- Scatter plots with axis labels and titles (Mileage vs Price, Age vs Price)
- Clear print of training/test set sizes
- Side-by-side comparison of predicted vs actual prices
- R² score, model coefficients, and intercept printed for interpretability
- Illustrated best practices for avoiding data leakage

**Dataset:** `carprices.csv` — car mileage, age, and sell price

**Use Case:** Any supervised learning workflow where proper evaluation is required.

---

### 6. Multivariate Linear Regression

**File:** `multivariate_linear_regression.ipynb`

**Description:** Demonstrates multivariate linear regression using multiple independent variables to predict target values. Includes two real-world examples with comprehensive data preprocessing.

**Key Features:**
- Multiple feature regression (area, bedrooms, age → price)
- Advanced data preprocessing techniques
- Handling missing values with statistical methods
- Text-to-number conversion using word2number library
- Model coefficient interpretation
- Performance evaluation with R² score
- Real-world prediction examples

**Examples Covered:**
1. **Home Price Prediction:** Predicting house prices based on area, number of bedrooms, and age
2. **Salary Prediction:** Predicting candidate salaries based on experience, test scores, and interview performance

**Use Case:** Complex prediction tasks with multiple influencing factors (e.g., real estate pricing, salary estimation, sales forecasting)

**Formula:** `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`

---

### 7. Gradient Descent

**File:** `gradient_descent.ipynb`

**Description:** Custom implementation of the gradient descent optimization algorithm for linear regression. This demonstrates how machine learning algorithms learn from data by iteratively updating parameters to minimize error.

**Key Features:**
- From-scratch implementation without using sklearn's optimization
- Configurable learning rate and iteration count
- Parameter history tracking for convergence visualization
- Cost function (MSE) calculation at each iteration
- Comparison with sklearn's LinearRegression for validation
- Two complete examples showing learning rate scaling

**Examples Covered:**
1. **Simple Dataset (y = 2x + 3):**
   - Learning rate: 0.08
   - Demonstrates basic gradient descent convergence
   - Visualization of fitted line
   
2. **Test Scores Dataset (Math vs CS):**
   - Learning rate: 0.0001 (adjusted for larger values)
   - Real-world application with comparison to sklearn
   - Shows importance of scaling learning rate with data magnitude

**Key Concepts:**
- **Cost Function:** Mean Squared Error (MSE) = (1/n) × Σ(y - ŷ)²
- **Gradient Calculation:** Partial derivatives with respect to m and b
- **Parameter Update:** θ = θ - α × ∇J(θ)
  - m = m - learning_rate × ∂cost/∂m
  - b = b - learning_rate × ∂cost/∂b

**Important Learning:**
⚠️ **Learning Rate Scaling** - The learning rate must be adjusted based on your data scale:
- Small values (1-10): Use larger learning rate (e.g., 0.08)
- Large values (50-100): Use smaller learning rate (e.g., 0.0001)
- Too large → Parameters explode (NaN values)
- Too small → Convergence takes too long

**Use Case:** Understanding the fundamentals of how ML algorithms optimize parameters, custom optimization needs, educational purposes

**Dataset:** Includes `test_scores.csv` with sample student data

---

### 8. Decision Tree (Salary Prediction)

**File:** `decision_tree.ipynb` (salary example)

**Description:** Uses a decision tree classifier to predict whether a salary is above a threshold (`salary_more_then_100k`) based on company, job title, and education degree.

**Key Features:**
- Label encoding of categorical variables (`company`, `job`, `degree`)
- Creation of numeric feature matrix for tree input
- Training and scoring a `DecisionTreeClassifier`
- Prediction examples for different encoded feature combinations
- Dedicated markdown explaining key tree hyperparameters:
  - `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `random_state`

**Use Case:** Classification problems with categorical predictors and interpretable tree structure.

---

### 9. Decision Tree (Titanic Survival)

**File:** `decision_tree.ipynb` (Titanic exercise)

**Description:** Decision tree model built on the classic Titanic dataset to predict passenger survival using cleaned and encoded features.

**Key Features:**
- Feature selection and dropping of high-cardinality text fields
- Missing values in `Age` and `Embarked` filled **before** label encoding (important: `LabelEncoder` cannot handle NaN)
- Label encoding for `Sex` and `Embarked`
- Train/test split with `random_state=42` for reproducibility
- Training a `DecisionTreeClassifier` for survival prediction
- Accuracy score on the hold-out test set

**Use Case:** Tabular classification with a mix of numeric and categorical features, using a well-known benchmark dataset.

## 📈 Example Outputs

### Linear Regression
- Learns the relationship: `Score = Slope × Hours + Intercept`
- Provides regression line visualization
- Calculates accuracy metrics

### Logistic Regression
- Estimates probability of passing/failing
- Visualizes sigmoid curve and decision boundary
- Shows probability distribution for predictions

### Multivariate Linear Regression
- Predicts target using multiple features simultaneously
- Shows individual impact of each feature (coefficients)
- Handles complex real-world datasets with missing values
- Provides interpretable model equation
- Demonstrates data preprocessing best practices

### Gradient Descent
- Shows step-by-step parameter optimization process
- Visualizes convergence of cost function over iterations
- Tracks evolution of slope (m) and intercept (b) parameters
- Compares custom implementation with sklearn's results
- Demonstrates impact of learning rate on training
- Provides convergence plots and fitted line visualization

---

### 10. Support Vector Machine (SVM) — Iris Classification

**File:** `SVM.ipynb`

**Description:** Applies Support Vector Machine classification on the classic Iris dataset to distinguish between three flower species (setosa, versicolor, virginica) using sepal and petal measurements.

**Key Features:**
- Loads the built-in `sklearn` Iris dataset and builds a labeled DataFrame
- Scatter plots comparing sepal and petal dimensions across species
- Train/test split (80/20) with `random_state=42`
- `SVC` model with `kernel='linear'`, `C=10`, `gamma=10`
- Accuracy score on the test set (~96.7%)
- Single-sample prediction example
- **Exercise included:** Train SVM on the digits dataset with `rbf` and `linear` kernels, tune `C` and `gamma` for best accuracy

**Dataset:** Built-in `sklearn` Iris dataset (150 samples, 4 features, 3 classes)

**Use Case:** Multiclass classification on numeric features; understanding kernel-based separation of classes.

---

### 11. Random Forest — Digits & Titanic

**File:** `Random_forest.ipynb`

**Description:** Applies Random Forest classification on two datasets: the `sklearn` digits dataset (handwritten digit recognition) and the Titanic survival dataset.

**Key Features:**
- Loads and visualizes the digits dataset (8×8 pixel images)
- Builds a DataFrame from the 64 pixel features + target label
- Train/test split (80/20) for digits classification
- **Titanic section:** loads `titanic.csv`, handles missing `Age` and `Embarked` values, applies one-hot encoding (`pd.get_dummies`) for `Sex` and `Embarked`
- `RandomForestClassifier(n_estimators=60, max_depth=5, random_state=42)` trained on Titanic data
- Accuracy score on the test set (~81.6%)
- Feature importance ranking printed — `Sex_male` is the most important feature (43.4%)

**Dataset:** Built-in `sklearn` digits dataset + `titanic.csv`

**Use Case:** Ensemble learning for both image classification and tabular survival prediction; understanding feature importance in Random Forests.

---

## 🛠️ Technologies Used

- **Python** - Programming language
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Matplotlib** - Data visualization
- **scikit-learn** - Machine learning library

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements!

## 📧 Contact

For questions or suggestions, please open an issue in this repository.

---

**Happy Learning! 🎓**
