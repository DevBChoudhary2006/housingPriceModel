# housingPriceModel
A Kaggle competition that focuses on making a model 
(note it is better visible within the kaggle competition here: https://www.kaggle.com/code/devbchoudhary/house-price-competion)

# House Prices: Advanced Regression Techniques

Predicts home sale prices (Ames, Iowa) from 79 features for the Kaggle House Prices competition.

## Approach

1. **EDA**: `SalePrice` distribution and key feature histograms.
2. **Preprocessing**: median impute + scale for numeric columns; mode impute + one-hot for categorical.
3. **Target**: models train on `log1p(SalePrice)` and predictions are converted back with `expm1`.
4. **Models**: Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, XGBoost, LightGBM, compared with 5-fold CV.
5. **Feature engineering**: total square footage, total bathrooms, age at sale, remodel age, and quality interactions.
6. **Tuning**: `RandomizedSearchCV` on XGBoost.
7. **Submission**: tuned XGBoost refit on all training data.

**Metric:** RMSE on log price (lower is better).

## Setup
bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm
