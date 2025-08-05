# 🏠 Airbnb Price Prediction with Machine Learning

This project was completed as part of the Break Through Tech AI Program’s Fall 2024 AI Studio. The goal was to explore a real-world dataset and apply the full machine learning life cycle — from data preprocessing to model evaluation — to solve a predictive task.

## 📌 Problem Statement

**Can we predict the nightly price of an Airbnb listing based on listing details such as room type, number of bedrooms, and review scores?**

We used the NYC Airbnb listings dataset to build regression models that help estimate listing prices. This can support Airbnb's smart pricing recommendations, improve host decision-making, and enhance guest trust through price transparency.

---

## 🧠 Machine Learning Life Cycle Overview

### 🔍 Step 1: Define the ML Problem
- **Label:** `price` (nightly listing price)
- **Type:** Supervised learning → Regression
- **Features:**  
  - `room_type`, `accommodates`, `bathrooms`, `bedrooms`, `beds`
  - `number_of_reviews`, `review_scores_rating`
  - `host_is_superhost`
  - `amenities_count` (engineered)
  - Binary indicators for missing values: `*_is_null`

---

### 📊 Step 2: Exploratory Data Analysis (EDA)
- Missing values in `bedrooms` and `beds` were imputed with median values.
- Binary columns created to capture null indicators.
- One-hot encoding used for categorical variables like `room_type` and `host_is_superhost`.
- A new numeric feature `amenities_count` was engineered to count listed amenities.
- Visualizations (Seaborn & Matplotlib) used to inspect distributions and outliers.

---

### 🏗️ Step 3: Feature Preparation
**Final Feature Set:**
```python
[
    'accommodates', 'bathrooms', 'bedrooms', 'beds',
    'number_of_reviews', 'review_scores_rating',
    'bedrooms_is_null', 'beds_is_null', 'amenities_count',
    'room_type_Private room', 'room_type_Shared room',
    'host_is_superhost_True'
]
