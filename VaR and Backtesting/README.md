# Financial Data Science

**Developed by Reza Zamani, Hrafnhildur Líf, and Paraj as part of the Financial Data Science program, under the supervision of Ali Kakhbod.**

Applied financial data science and machine-learning projects focused on stock-return prediction, panel-data modeling, ensemble learning, neural networks, patent analytics, and image-based financial forecasting.

This repository contains a collection of projects developed around predictive modeling problems in finance and economics. The projects emphasize out-of-sample prediction, careful validation, feature engineering, model comparison, reproducibility, and the application of modern machine-learning methods to structured and unstructured financial data.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Cross-Sectional Stock Return Prediction](#1-cross-sectional-stock-return-prediction)
3. [Panel Stock Return Prediction](#2-panel-stock-return-prediction)
4. [Patent Citation Prediction](#3-patent-citation-prediction)
5. [Image-Based Stock Prediction with CNNs](#4-image-based-stock-prediction-with-cnns)
6. [Methods and Technologies](#methods-and-technologies)
7. [Core Data-Science Themes](#core-data-science-themes)
8. [Repository Structure](#repository-structure)
9. [Skills Demonstrated](#skills-demonstrated)
10. [Reproducibility](#reproducibility)

---

# Project Overview

This repository covers four major predictive-modeling projects in financial data science.

| Project | Problem Type | Target | Main Data Type |
|---|---|---|---|
| Cross-Sectional Stock Return Prediction | Regression | Stock return | Firm characteristics |
| Panel Stock Return Prediction | Regression | Daily stock return | Firm × time panel |
| Patent Citation Prediction | Regression | Future patent citations | Patent characteristics |
| Image-Based Stock Prediction | Classification & Regression | 5-day stock returns | Financial chart images |

The projects progress from structured financial datasets to ensemble learning, neural networks, and computer-vision-based financial forecasting.

---

# 1. Cross-Sectional Stock Return Prediction

## Problem Set 1 — Task 1

### Objective

Develop predictive models for stock returns using firm-level financial characteristics.

The dataset is purely cross-sectional: each firm is observed once, with no time dimension. The objective is to learn the relationship between firm characteristics and a single-period stock return and then generate predictions for a held-out test sample.

### Target Variable

`ret`

Single-period stock return.

### Input Features

The predictive variables include:

- `firm_id` — firm identifier
- `size` — standardized log market capitalization
- `value` — book-to-market characteristic
- `profit` — profitability
- `invest` — investment / asset growth
- `mom` — past 12-month momentum
- `rating` — categorical credit-rating bucket

Credit-rating categories include:

- AAA
- AA
- A
- BBB
- BB
- B

### Data Structure

Two datasets are used:

- Training data containing both features and the target return
- Testing data containing only explanatory variables

The target for the test set is withheld, making model selection and out-of-sample validation especially important.

### Modeling Workflow

The project follows a complete predictive-modeling pipeline:

1. Load and inspect the financial data
2. Examine feature distributions and data quality
3. Identify numerical and categorical variables
4. Encode categorical information
5. Construct training and validation samples
6. Train predictive models
7. Compare validation performance
8. Tune model specifications
9. Select the preferred model
10. Retrain using the available training sample
11. Generate predictions for the held-out firms
12. Export the final prediction file

### Key Challenges

- High idiosyncratic noise in stock returns
- Cross-sectional heterogeneity
- Categorical feature handling
- Model-selection uncertainty
- Generalization to unseen firms

### Prediction Output

The final prediction file has the structure:

```text
firm_id,y_hat
```

where:

- `firm_id` identifies the firm
- `y_hat` is the predicted stock return

### Deliverable

`task1_predictions.csv`

---

# 2. Panel Stock Return Prediction

## Problem Set 1 — Task 2

### Objective

Predict daily stock returns using a panel dataset containing firm-level and macroeconomic information.

Unlike the first project, observations vary across both firms and time.

### Dataset Structure

The panel contains:

- 100 firms
- 200 business days
- Approximately 20,000 firm-day observations

### Target Variable

`ret`

Daily stock return.

### Input Features

The dataset includes:

- `date` — trading date
- `firm_id` — firm identifier
- `macro1` — categorical business-cycle regime
- `macro2` — continuous macroeconomic index
- `price` — stock price
- `firm1` — firm-level characteristic
- `firm2` — firm-level characteristic
- `firm3` — firm-level characteristic

### Macroeconomic Regimes

The categorical macroeconomic variable contains business-cycle states such as:

- Expansion
- Contraction
- Recovery
- Peak
- Trough

### Key Modeling Challenge

Financial panel prediction requires careful treatment of the time dimension.

Models intended to forecast future returns should be evaluated using historical information to predict later observations.

This makes chronological validation particularly important and reduces the risk of look-ahead bias.

### Modeling Workflow

The project includes:

1. Data loading and inspection
2. Panel-data organization
3. Date preprocessing
4. Firm-level feature analysis
5. Macroeconomic feature analysis
6. Categorical-variable encoding
7. Time-aware validation
8. Predictive-model training
9. Hyperparameter tuning
10. Out-of-sample model comparison
11. Final model estimation
12. Firm-day return prediction
13. Prediction-file generation

### Validation Considerations

Special attention is given to:

- Chronological splitting
- Preventing future information from entering training data
- Avoiding preprocessing leakage
- Maintaining consistent preprocessing between training and testing samples
- Evaluating genuine out-of-sample predictive performance

### Prediction Output

The final prediction file has three columns:

```text
firm_id,date,y_hat
```

where:

- `firm_id` identifies the stock
- `date` identifies the trading day
- `y_hat` is the predicted daily return

### Deliverable

`task2_predictions.csv`

---

# 3. Patent Citation Prediction

## Problem Set 2 — Problem 4

### Objective

Predict the number of citations that a patent will receive.

The target variable is:

`fcitALL`

This project extends the predictive-modeling framework beyond traditional market data and examines whether patent characteristics can be used to predict future technological impact.

### Modeling Framework

Three separate model classes are developed.

## Random Forest

A Random Forest model is trained using non-text patent characteristics.

Random Forests are useful for capturing:

- Nonlinear relationships
- Feature interactions
- Threshold effects
- Heterogeneous patterns across observations

## XGBoost

An XGBoost model is trained using non-text patent covariates.

Gradient boosting provides a flexible framework for improving predictive performance by sequentially learning from the residual errors of previous trees.

## Neural Network

A neural-network model provides a third predictive approach.

The neural-network framework also provides flexibility to incorporate additional information, including text-based information when appropriate.

### Modeling Workflow

The project includes:

1. Patent-data loading
2. Data cleaning
3. Feature inspection
4. Feature selection
5. Missing-value treatment
6. Training and validation design
7. Random Forest modeling
8. XGBoost modeling
9. Neural-network modeling
10. Hyperparameter selection
11. Model comparison
12. Out-of-sample patent citation prediction
13. Prediction-file generation

### Target

```text
fcitALL
```

Number of citations received by a patent.

### Models

- Random Forest
- XGBoost
- Neural Network

### Output

Each model produces predictions associated with the patent identifier.

Conceptually:

```text
patent_id,predicted_citations
```

### Main Learning Objective

This project provides a direct comparison between:

- Bagging
- Gradient boosting
- Neural-network modeling

for a common out-of-sample prediction problem.

---

# 4. Image-Based Stock Prediction with CNNs

## Problem Set 2 — Problem 5

### Objective

Apply computer-vision methods to financial-market prediction by transforming historical stock-price information into images and training Convolutional Neural Networks to extract predictive patterns.

Instead of supplying conventional numerical time-series features directly to the model, historical market information is represented visually through stock-price charts.

CNNs are then used to learn potentially predictive graphical patterns from these images.

## Financial Assets

The analysis covers four major U.S. technology companies:

- Apple
- Microsoft
- Google
- Amazon

## Historical Sample

Stock-price data span:

```text
2005-01-01 to 2021-12-31
```

The primary daily market variables include:

- Open
- High
- Low
- Close

## Financial Image Construction

Historical stock-price information is transformed into chart images.

The chart-generation process allows flexibility in:

- Historical window length
- Image resolution
- Chart dimensions
- Price representation
- Scaling
- Visual structure
- Additional financial information

Each image represents information available up to a particular trading date.

The corresponding prediction target is calculated using future stock returns.

## Task A — Five-Day Return Direction Classification

### Objective

Predict whether the stock return over the following five trading days will be:

- Positive
- Negative

The target is based on the close-to-close return over the next five trading days.

### Machine-Learning Formulation

This is a binary image-classification problem.

The target can be represented as:

\[
Y_t =
egin{cases}
1, & R_{t,t+5} > 0 \
0, & R_{t,t+5} \leq 0
\end{cases}
\]

where:

- \(R_{t,t+5}\) is the future five-day stock return.

### CNN Workflow

1. Download and preprocess historical stock-price data
2. Generate stock-chart images
3. Construct future five-day directional labels
4. Match each image with its target
5. Divide the images chronologically into training and testing samples
6. Normalize and preprocess images
7. Build the CNN architecture
8. Train the classification model
9. Validate predictive performance
10. Save the final trained model
11. Reload the model for inference
12. Generate predictions for test images
13. Export prediction results

### Output

The classification prediction file contains:

```text
ticker,date,prediction
```

## Task B — Five-Day Log-Return Prediction

### Objective

Predict the continuous stock log return over the following five trading days.

The target can be represented as:

\[
r_{t,t+5}
=
\log
\left(
rac{P_{t+5}}{P_t}
ight)
\]

where:

- \(P_t\) is the current closing price
- \(P_{t+5}\) is the closing price five trading days later

### Machine-Learning Formulation

This is an image-based regression problem.

The CNN learns a mapping:

```text
Stock Chart Image
        ↓
Future 5-Day Log Return
```

### Workflow

1. Construct historical chart images
2. Calculate future five-day log returns
3. Match each image with its return target
4. Divide observations chronologically
5. Train the CNN regression model
6. Evaluate out-of-sample predictions
7. Save the trained model
8. Reload the model for inference
9. Generate test predictions
10. Export prediction results

### Output

The regression prediction file contains:

```text
ticker,date,predicted_log_return
```

## Train / Test Design for the CNN Project

The chronological split is:

### Training Period

```text
2005-01-01 through 2018-12-31
```

### Testing Period

```text
2019-01-01 through 2021-12-31
```

The chronological structure is designed to ensure that future observations are not used to train models intended to predict earlier dates.

## CNN Project Deliverables

The complete image-based stock-prediction project includes:

- Classification notebook
- Regression notebook
- Classification model
- Regression model
- Test prediction CSV for classification
- Test prediction CSV for regression
- Sample test images
- Reproducible model-loading and inference pipeline

The project emphasizes reproducibility so that trained models can be loaded and used to generate predictions without retraining the entire network.

---

# Methods and Technologies

## Financial Data Science

- Stock Return Prediction
- Cross-Sectional Asset Prediction
- Panel Financial Data
- Firm Characteristics
- Macroeconomic Variables
- Financial Forecasting
- Out-of-Sample Evaluation
- Chronological Validation
- Feature Engineering
- Predictive Modeling

## Machine Learning

- Regression
- Classification
- Decision Trees
- Random Forest
- Gradient Boosting
- XGBoost
- Ensemble Learning
- Neural Networks
- Convolutional Neural Networks

## Deep Learning

- Feedforward Neural Networks
- Image Classification
- Image Regression
- Convolutional Feature Extraction
- Model Training
- Validation
- Saved-Model Inference

## Data Types

The projects work with several types of data:

- Cross-sectional financial data
- Panel financial data
- Time-series data
- Macroeconomic data
- Firm-level characteristics
- Patent data
- Financial chart images
- Numerical features
- Categorical features

## Python Ecosystem

The projects use the Python data-science and machine-learning ecosystem, including tools for:

- Data manipulation
- Numerical computing
- Statistical analysis
- Machine learning
- Deep learning
- Visualization
- Model evaluation

Typical libraries include:

```text
Python
NumPy
pandas
scikit-learn
XGBoost
PyTorch / TensorFlow
Matplotlib
Jupyter Notebook
```

---

# Core Data-Science Themes

## 1. Out-of-Sample Prediction

The primary objective is not simply to fit historical observations.

Models are evaluated according to their ability to predict previously unseen data.

This is particularly important in financial applications, where in-sample model fit does not necessarily translate into useful predictive performance.

## 2. Proper Validation

Financial data require careful construction of training and validation samples.

For time-dependent datasets, chronological validation helps prevent future information from leaking into model training.

The general structure is:

```text
Past Data
    ↓
Model Training
    ↓
Future Data
    ↓
Out-of-Sample Evaluation
```

## 3. Avoiding Look-Ahead Bias

A major concern in financial machine learning is accidental use of information that would not have been available at the time of prediction.

Potential sources include:

- Random splitting of time-dependent observations
- Full-sample preprocessing
- Future-derived features
- Improper normalization
- Target leakage

The projects emphasize maintaining a realistic forecasting framework.

## 4. Feature Engineering

The projects use several forms of predictive information:

- Firm characteristics
- Stock prices
- Macroeconomic conditions
- Business-cycle regimes
- Patent characteristics
- Financial chart images

Feature engineering plays an important role in transforming these inputs into representations suitable for predictive modeling.

## 5. Model Comparison

Different model classes capture different forms of predictive structure.

The projects compare approaches involving:

- Statistical prediction
- Tree-based machine learning
- Ensemble methods
- Gradient boosting
- Neural networks
- Convolutional neural networks

## 6. Structured and Unstructured Data

The repository demonstrates predictive modeling using both:

### Structured Data

Examples:

- Firm characteristics
- Macroeconomic variables
- Stock prices
- Patent covariates

### Unstructured / Image-Based Data

Examples:

- Historical stock-price chart images

This allows comparison between conventional financial machine learning and deep-learning approaches.

## 7. Reproducibility

A key goal throughout the projects is to maintain a reproducible workflow.

The general pipeline is:

```text
Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Training / Validation
   ↓
Model Estimation
   ↓
Model Evaluation
   ↓
Model Selection
   ↓
Final Model
   ↓
Out-of-Sample Prediction
   ↓
Prediction File
```

---

# Repository Structure

A suggested organization for this repository is:

```text
financial-data-science/
│
├── README.md
│
├── ps1/
│   │
│   ├── task1_cross_sectional_return_prediction/
│   │   ├── notebooks/
│   │   ├── data/
│   │   ├── outputs/
│   │   └── README.md
│   │
│   └── task2_panel_return_prediction/
│       ├── notebooks/
│       ├── data/
│       ├── outputs/
│       └── README.md
│
├── ps2/
│   │
│   ├── patent_citation_prediction/
│   │   ├── random_forest/
│   │   ├── xgboost/
│   │   ├── neural_network/
│   │   ├── outputs/
│   │   └── README.md
│   │
│   └── cnn_stock_prediction/
│       ├── classification/
│       ├── regression/
│       ├── images/
│       ├── models/
│       ├── outputs/
│       └── README.md
│
├── models/
│
└── outputs/
```

The folder structure can be adapted to match the actual notebooks, data, models, and prediction files contained in the repository.

---

# Project Summary

## Problem Set 1

### Task 1 — Cross-Sectional Return Prediction

Predict stock returns from firm characteristics using cross-sectional financial data.

### Task 2 — Panel Return Prediction

Predict daily stock returns using a panel containing firm-level and macroeconomic information.

## Problem Set 2

### Problem 4 — Patent Citation Prediction

Predict future patent citations using:

- Random Forest
- XGBoost
- Neural Network

### Problem 5 — CNN-Based Stock Prediction

Transform financial price histories into images and use CNNs for:

- Five-day stock-return direction classification
- Five-day stock log-return regression

---

# Skills Demonstrated

This repository demonstrates practical experience in:

- Financial Data Science
- Quantitative Research
- Financial Machine Learning
- Predictive Modeling
- Stock Return Forecasting
- Cross-Sectional Analysis
- Panel-Data Modeling
- Macroeconomic Feature Modeling
- Time-Aware Validation
- Out-of-Sample Testing
- Feature Engineering
- Decision Trees
- Random Forest
- Gradient Boosting
- XGBoost
- Neural Networks
- Deep Learning
- Computer Vision
- CNN-Based Financial Forecasting
- Model Evaluation
- Model Selection
- Reproducible Research
- Python-Based Data Analysis

---

# Reproducibility

Reproducibility is an important component of the repository.

Each project is designed to contain sufficient code to reproduce:

1. Data preprocessing
2. Feature construction
3. Training / validation splitting
4. Model estimation
5. Model evaluation
6. Final model selection
7. Out-of-sample predictions
8. Exported prediction files

For deep-learning projects, trained models should also be loadable independently from the training process so that predictions can be reproduced without retraining the entire network.

---

# Notes

Some datasets used in these projects may be course-provided, proprietary, restricted, or too large to distribute publicly.

When raw datasets cannot be included in the repository, the notebooks and source code document the expected input format and modeling workflow so that the analysis can be reproduced once the required data are available.

---

# About

This repository focuses on the intersection of:

**Financial Data Science · Quantitative Finance · Machine Learning · Deep Learning**

with an emphasis on applying modern predictive methods to financial, economic, patent, and alternative datasets.

---

## Contributors

**Reza Zamani**  
**Hrafnhildur Líf**  
**Paraj**

### Supervision

**Ali Kakhbod**

Financial Data Science Program
