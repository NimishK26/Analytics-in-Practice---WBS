# Analytics-in-Practice---WBS
Predictive modelling project for an eCommerce platform using machine learning to classify customer review sentiment based on transaction and product data.

# Nile eCommerce Review Prediction

A data science project to develop and evaluate machine learning models that predict the likelihood of customers leaving positive reviews on a South American eCommerce platform, Nile.

## Overview

This project was developed as part of the Analytics in Practice module at the University of Warwick. The goal is to support business decision-making by identifying which customers are likely to leave positive reviews (4–5 stars). The resulting insights enable resource-efficient targeting for customer engagement and reputation management.

## Business Context

Nile, a major South American eCommerce company, seeks to enhance its reputation by increasing the volume of genuine, positive customer reviews. The company provided a real-world relational dataset exported from its internal systems. This project builds a predictive model and provides actionable recommendations to optimize customer outreach strategies.

## Dataset

The dataset consists of 8 CSV files and a product category translation table. These were joined to form a unified dataset for analysis. Key tables include:

- Orders
- Order Reviews
- Customers
- Sellers
- Products
- Geolocation
- Payments
- Order Items
- Product Category Names

## Project Structure

- `IB9BW0 41 Python.py`: Python script containing the entire data pipeline and modeling code.
- `IB9BW0 41 Technical Report.pdf`: In-depth technical documentation and evaluation of the models.
- `IB9BW0 41 PPT.pdf`: Presentation slides used to pitch the project to a simulated client board.
- `data/`: Directory for raw CSV files (not included in the repo due to size and privacy).
- `README.md`: This documentation file.

## Methodology

The project followed the CRISP-DM methodology:

1. **Business Understanding**: Define business goals and key success criteria.
2. **Data Understanding**: Explore data structures and identify quality issues.
3. **Data Preparation**: Clean data, perform feature engineering, and create binary classification labels.
4. **Modeling**: Use tree-based classifiers (Random Forest, GBDT, XGBoost).
5. **Evaluation**: Assess performance using precision, recall, F1-score, and AUC.
6. **Deployment Recommendations**: Provide guidance on real-world integration and scalability.

## Feature Engineering

Key engineered features included:

- Time gaps between events (purchase, delivery, review).
- Product description length.
- Delivery vs. estimated time.
- Freight cost ratios.
- Aggregated customer behavior (e.g., total purchases).

## Models and Results

### Models Used:
- Random Forest
- Gradient Boosted Decision Trees (GBDT)
- XGBoost

### Performance Metrics (Test Set - Random Forest):
- **Precision**: 0.848
- **Recall**: 0.681
- **F1 Score**: 0.716
- **Accuracy**: 83.97%

SMOTE was applied to address class imbalance in the training set. Hyperparameter tuning was conducted using both RandomizedSearchCV and Bayesian Optimization.

## Key Insights

- Reviews are highly influenced by delivery experience and communication timing.
- Random Forest achieved the best balance of precision and interpretability.
- The model can help reduce marketing inefficiencies by targeting likely reviewers.

## Recommendations

- Focus on top predictive features (e.g., delivery delays, product descriptions).
- Implement targeted incentives for high-likelihood reviewers.
- Continuously monitor and retrain the model with new data.
- Explore text-based sentiment models for further improvement.

