# Insurance Marketing Optimization Propensity Model

## Overview
This project focuses on building a propensity model to optimize marketing efforts for an insurance company. The goal is to identify potential customers who are likely to engage with specific actions, such as purchasing insurance policies. By leveraging data analysis and machine learning techniques, we aim to improve the efficiency and effectiveness of the company's marketing campaigns.

## Problem Statement
The insurance company faces challenges in identifying and targeting potential customers effectively. Despite having access to large datasets, the company struggles to leverage this data to predict customer behavior accurately. The objective of this project is to develop a propensity model that can forecast the likelihood of individuals becoming customers. By doing so, the company can optimize its marketing strategies and allocate resources more efficiently.

## Data
The dataset provided by the insurance company consists of historical data (train.xlsx) and a list of potential customers (test.xlsx). The data includes various features such as customer demographics, previous interactions, and marketing campaign outcomes. The target variable is whether a customer responded positively to the marketing campaign (1 for yes, 0 for no).

## Project Structure
The project is organized into the following directories:

- **notebooks**: Contains Jupyter notebooks for exploratory data analysis (EDA) and model building.
  - `exploratory_analysis.ipynb`: Notebook for analyzing and understanding the dataset.
  - `model_building.ipynb`: Notebook for building and evaluating machine learning models.

- **pipeline**: Includes Python scripts for data preprocessing, model training, and model evaluation.
  - `preprocess.py`: Script for preprocessing the dataset, handling missing values, and feature engineering.
  - `train.py`: Script for training the machine learning model using the preprocessed data.
  - `evaluate.py`: Script for evaluating the trained model's performance on test data.

- **data**: Stores raw and processed datasets, along with a README file for data documentation.
  - **raw**: Contains raw data files provided by the insurance company.
    - `train.xlsx`: Historical data for model training.
    - `test.xlsx`: List of potential customers for marketing.
  - **processed**: Contains processed data files generated during preprocessing.
    - `train_preprocessed.xlsx`: Preprocessed training data.
    - `testingCandidate.csv`: Preprocessed test data with predictions.
  - `README.md`: Documentation for data files, including descriptions and usage instructions.

- **models**: Contains the trained machine learning model (ensemble_classifier_model.pkl).

- **visuals**: Contains visualization outputs generated during the analysis.

- **README.md**: Main documentation file providing an overview of the project, instructions, and project structure details.

## Features Involved
The project involves the following key activities:

1. **Exploratory Data Analysis (EDA)**:
2. **Data Cleaning**:
3. **Dealing with Imbalanced Data**:
4. **Feature Engineering**:
5. **Model Selection**:
6. **Model Training and Evaluation**:
7. **Hyperparameter Tuning**:
8. **Model Deployment**:

## Data Access
The dataset for this project can be accessed [here](data/raw/train.xlsx) and [here](data/raw/test.xlsx).
