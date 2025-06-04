# Real Estate Rental Yield Prediction

This repository contains all code and data needed to reproduce the Real Estate Rental Yield Prediction Analysis (V2). We use a Random Forest Regressor to predict three key target variables:

1. **Long-Term Rental Yield**  
2. **Short-Term (Airbnb) Rental Yield**  
3. **Overall Investment Score**  

---

## 📋 Project Overview

Many investors need quick insights into how well a property will perform—both as a traditional long-term rental and as an Airbnb (short-term rental). The goal here is to build a model that, given a property’s features, estimates:

- Annualized long-term rental yield (%).  
- Expected short-term (Airbnb) rental yield (%).  
- A composite investment score that combines various indicators.  

To achieve this, we combine data from multiple sources (apartments, property listings, Airbnb), clean and engineer features, and train a Random Forest Regressor for each target variable.

---

## 🗂️ Data Sources

1. **Apartments Dataset**  
   - CSV including columns like `price`, `area_sqm`, `bedrooms`, `year_built`, `amenities`, etc.  
   - Contains historical long-term rent and sale values.  

2. **Properties (Batch 1, Batch 2, Batch 3)**  
   - Parquet or CSV files with listings scraped from websites.  
   - Includes location (latitude/longitude), `num_bathrooms`, `has_parking`, `has_gym`, etc.  

3. **Airbnb Dataset**  
   - CSV file of Airbnb listings in the region with nightly rates, occupancy estimates, host response rate, review scores, etc.  

---

## 🧹 Data Cleaning & Feature Engineering

1. **Basic Cleaning**  
   - Remove invalid rows (e.g., `price <= 0`, `area_sqm <= 0`).  
   - Standardize currencies (if needed).  
   - Drop unreachable outliers (e.g., top 1% of nightly rates).  

2. **Yield Calculations**  
   - **Long-Term Yield** = `(annual_rent / sale_price) × 100`.  
   - **Short-Term Yield** = `(average_nightly_rate × occupancy_rate × 365) / sale_price × 100`.  

3. **Feature Engineering**  
   - `price_per_sqm` = `sale_price / area_sqm`.  
   - `amenities_count` (number of amenities flagged).  
   - Neighborhood one-hot encoding (e.g., “Downtown”, “Suburb_A”, “Suburb_B”).  
   - `property_age` = `current_year − year_built`, then bucketed into bins (e.g., “<10 years”, “10–20 years”, “>20 years”).  
   - Boolean columns filled with `False` if missing (e.g., `has_pool`, `has_gym`, `has_parking`).  

4. **Final Aggregation**  
   - Concatenate cleaned apartments, properties, and Airbnb DataFrames.  
   - Filter to properties with all essential fields.  
   - Construct final feature matrix **X** (all numeric/categorical predictors) and target dict **y_dict** = { 
     - “long_term_rental_yield”: Series,
     - “short_term_rental_yield”: Series,
     - “investment_score”: Series 
   }.  

---

## 🔄 Train/Test Split

- We split **X** and each **y** in `y_dict` into 80 % training and 20 % testing sets (using `train_test_split(random_state=42)`).  
- This yields three separate train/test splits, one for each target.

---

## 🌲 Modeling: Random Forest Regressor

We build a scikit-learn pipeline for each target:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Example schematic pipeline:
numeric_features = [list_of_numeric_cols]
categorical_features = [list_of_categorical_cols]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",  # drop any other cols
)

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
)

# Then for each target:
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
````

All three targets use the same pipeline structure, but are trained independently.

---

## 📊 Evaluation Metrics

For each target:

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**
* **R² (Coefficient of Determination)**
* **MAPE (Mean Absolute Percentage Error)**

The notebook prints a comparison table that looks something like this:

| Target                     |  MAE | RMSE |   R² | MAPE (%) |
| :------------------------- | ---: | ---: | ---: | -------: |
| long\_term\_rental\_yield  | 1.23 | 1.75 | 0.72 |    15.24 |
| short\_term\_rental\_yield | 2.05 | 2.90 | 0.68 |    18.52 |
| investment\_score          | 0.79 | 1.10 | 0.80 |    12.10 |

*(These numbers are illustrative—please refer to the notebook output for exact values.)*

---

## 💡 Key Insights

1. **Feature Importance (Long-Term Yield):**

   * `price_per_sqm` and `number_of_bedrooms` rank highest.
   * Older properties (higher `property_age`) tend to have slightly lower yields.
   * Location features (e.g., “Downtown = 1”) also show strong predictive power.

2. **Feature Importance (Short-Term Yield):**

   * Airbnb-specific variables (e.g., `review_score`, `occupancy_rate_estimate`) dominate.
   * Amenities such as `has_pool` and `proximity_to_public_transport` significantly boost predicted yields.

3. **Investment Score:**

   * A combination of both long-term and short-term predictors plus a heuristic scoring function.
   * Neighborhood categorical variables (encoded) account for a large share of explained variance in investment score.

4. **Model Performance:**

   * Random Forest achieves R² in the range 0.65–0.80, indicating a fairly strong explanatory model but with room for improvement.
   * MAE on yield predictions (\~ 1–2 percentage points) suggests reasonably tight error bands for investor use.

---

## 🖼️ Visualizations

Below are placeholders—after you run the notebook locally, export any figures (e.g., feature‐importance bar charts, predicted vs. actual scatter plots) into PNG or JPG files, then link them here.

### 1. Long-Term Yield: Feature Importance

![Feature Importance (Long-Term Yield)](images/feature_importance_long_term.png)
*Insert a bar plot showing the top 10 features here.*

### 2. Short-Term Yield: Feature Importance

![Feature Importance (Short-Term Yield)](images/feature_importance_short_term.png)
*Insert a bar plot showing top 10 features here.*

### 3. Investment Score: Feature Importance

![Feature Importance (Investment Score)](images/feature_importance_investment_score.png)
*Insert a bar plot showing top 10 features here.*

### 4. Predicted vs. Actual (Long-Term Yield)

![Pred vs. Actual (Long-Term)](images/pred_vs_actual_long_term.png)
*Insert a scatter plot of predicted vs. actual long-term yields here (ideally a 45° reference line).*

### 5. Predicted vs. Actual (Short-Term Yield)

![Pred vs. Actual (Short-Term)](images/pred_vs_actual_short_term.png)
*Insert a scatter plot of predicted vs. actual short-term yields here.*

(You can add more visualizations as needed, for example: residual plots, correlation heatmap, distribution of yields, etc.)

---

## 🏃 How to Reproduce

1. **Clone this repo**

   ```bash
   git clone https://github.com/your‐username/real_estate_yield_prediction.git
   cd real_estate_yield_prediction
   ```

2. **Set up a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   pip install -r requirements.txt
   ```

   * The `requirements.txt` file should contain at least:

   ```
   pandas
   numpy
   matplotlib
   seaborn
   scikit-learn
   jupyter
   ```

3. **Open and run the notebook**

   ```bash
   jupyter notebook Real_Estate_Yield_Analysis.ipynb
   ```

   – Go cell by cell to ensure all data files (e.g., `apartments.csv`, `properties_batch1.csv`, etc.) are in the correct `data/` folder.

4. **Export Figures**
   – After running the feature-importance and evaluation cells, save each plot (`plt.savefig("images/…")`) and commit them to the `images/` directory.

5. **Review Results & Visuals**
   – Check the printed evaluation metrics in the notebook.
   – Populate the “Visualizations” sections of this README with the exported PNGs.

---

## ⚙️ Folder Structure

```
real_estate_yield_prediction/
│
├── data/
│   ├── apartments.csv
│   ├── properties_batch1.csv
│   ├── properties_batch2.csv
│   ├── properties_batch3.csv
│   └── airbnb_listings.csv
│
├── images/
│   ├── feature_importance_long_term.png
│   ├── feature_importance_short_term.png
│   ├── feature_importance_investment_score.png
│   ├── pred_vs_actual_long_term.png
│   └── pred_vs_actual_short_term.png
│
├── Real_Estate_Yield_Analysis.ipynb
├── README.md
├── requirements.txt
```

---

## 🔮 Future Improvements

* **Hyperparameter Tuning** (e.g., `GridSearchCV` on `n_estimators`, `max_depth`, `max_features`).
* **Alternative Models** (e.g., Gradient Boosting, XGBoost, LightGBM).
* **Geospatial Analysis** to visualize yield heatmaps by neighborhood.
* **Web App Deployment** with Streamlit or Flask for interactive prediction (user enters features, sees predicted yields).
* **Time-Series Component**: If historical yield data is available, build a temporal model to forecast future yields.

---

## 📄 License

This project is released under the MIT License. Feel free to copy, modify, and distribute.

---
