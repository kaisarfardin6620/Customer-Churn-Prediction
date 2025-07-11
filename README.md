# Customer-Churn-Prediction

This repository contains a Jupyter Notebook (Churn_Prediction.ipynb) that implements various machine learning models to predict customer churn. The notebook covers data loading, exploratory data analysis (EDA), feature engineering, preprocessing, model training, evaluation, and hyperparameter tuning for a range of classification algorithms, including an Artificial Neural Network (ANN).

Table of Contents
Project Overview

Dataset

Features

Installation

Usage

Notebook Structure

Models Implemented

Results

Contributing

License

Project Overview
The main goal of this project is to build and compare different machine learning models to accurately predict whether a customer will churn (exit) or not. This is a common problem in many businesses, and effective churn prediction can help in retaining customers. The notebook explores various classification algorithms, handles data imbalances using SMOTE, and performs hyperparameter tuning to optimize model performance.

Dataset
The project uses the Churn_Modelling.csv dataset. This dataset contains information about bank customers, including their demographics, account details, and whether they have churned.

Key columns in the dataset include:

CreditScore: Credit score of the customer.

Geography: Country of the customer (France, Germany, Spain).

Gender: Gender of the customer.

Age: Age of the customer.

Tenure: Number of years the customer has been with the bank.

Balance: Account balance.

NumOfProducts: Number of products the customer uses.

HasCrCard: Whether the customer has a credit card (1 = Yes, 0 = No).

IsActiveMember: Whether the customer is an active member (1 = Yes, 0 = No).

EstimatedSalary: Estimated salary of the customer.

Exited: Whether the customer churned (1 = Yes, 0 = No) - Target Variable.

Features
The notebook performs the following feature engineering steps:

AgeGroup: Creates categorical age groups from the Age column.

Age_NumOfProducts_Interaction: An interaction feature combining Age and NumOfProducts.

The following columns are dropped as they are not relevant for modeling: RowNumber, CustomerId, Surname.

Installation
To run this notebook, you'll need to have Python installed along with several libraries. It's recommended to use a virtual environment.

Clone the repository (if applicable):

git clone <repository_url>
cd <repository_name>

Create a virtual environment:

python -m venv venv

Activate the virtual environment:

On Windows:

.\venv\Scripts\activate

On macOS/Linux:

source venv/bin/activate

Install the required packages:
The notebook uses several libraries. You can install them manually or create a requirements.txt file.

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost plotly tensorflow scikeras

Note: The ValueError: numpy.dtype size changed error observed in the output often indicates an incompatibility between numpy and scikit-learn versions. If you encounter this, try upgrading or downgrading numpy or scikit-learn to compatible versions.

Usage
Place the dataset: Ensure Churn_Modelling.csv is in the same directory as the Churn_Prediction.ipynb notebook, or update the path in the notebook accordingly.

Open the notebook:

jupyter notebook Churn_Prediction.ipynb

Run the cells: Execute the cells sequentially to perform data loading, EDA, preprocessing, model training, and evaluation.

Notebook Structure
The notebook is structured into several key sections:

Import Libraries: Imports all necessary Python libraries.

Load Data: Loads the Churn_Modelling.csv dataset.

Exploratory Data Analysis (EDA):

Displays basic information about the dataset (shape, head, tail, info, describe, null values, duplicates, column names, data types).

Visualizes distributions of numerical and categorical features.

Analyzes churn rate by various features (Age Group, Geography, Gender, Number of Products).

Examines relationships between numerical features using pair plots and scatter plots.

Generates a correlation matrix.

Uses Plotly for interactive visualizations.

Outlier Handling: Implements IQR-based outlier capping for numerical columns.

Feature Engineering: Creates new interaction features.

Data Preprocessing:

Drops irrelevant columns (RowNumber, CustomerId, Surname).

Splits data into training and testing sets.

Defines preprocessing pipelines for numerical (scaling) and nominal (one-hot encoding) features using ColumnTransformer and MinMaxScaler.

Addresses class imbalance using SMOTE within the pipelines.

Model Training and Evaluation:

Initial training and evaluation of various traditional machine learning models (Logistic Regression, Decision Tree, Random Forest, XGBoost, K-Nearest Neighbors, SVC, Gradient Boosting, AdaBoost, Bagging Classifier, Voting Classifier, Stacking Classifier).

Evaluates models based on Accuracy, Precision, Recall, F1-Score, ROC AUC, Confusion Matrix, and Classification Report.

Compares training and test accuracy to identify overfitting.

Plots ROC curves for all models.

Visualizes confusion matrices using Plotly.

Hyperparameter Tuning (GridSearchCV):

Defines hyperparameter grids for each model.

Performs GridSearchCV with StratifiedKFold cross-validation to find optimal hyperparameters.

Evaluates the performance of the tuned models.

Compares initial vs. tuned model performance.

Artificial Neural Network (ANN) Implementation:

Builds a basic Sequential ANN model.

Trains and evaluates the basic ANN model.

Implements manual hyperparameter tuning for the ANN using KerasClassifier and itertools.product with StratifiedKFold to find the best architecture and training parameters.

Trains the final tuned ANN model.

Models Implemented
The notebook trains and evaluates the following machine learning models:

Logistic Regression

Decision Tree

Random Forest

XGBoost

K-Nearest Neighbors (KNN)

Support Vector Classifier (SVC)

Gradient Boosting

AdaBoost

Bagging Classifier

Voting Classifier (Ensemble)

Stacking Classifier (Ensemble)

Artificial Neural Network (ANN)

Results
The notebook provides detailed performance metrics for each model, both before and after hyperparameter tuning. Visualizations like ROC curves and confusion matrices aid in understanding model behavior. The manual ANN tuning section specifically highlights the best hyperparameters found and the corresponding ROC AUC score.

Contributing
Feel free to fork this repository, explore the code, and suggest improvements. If you find any issues or have suggestions, please open an issue or submit a pull request.
