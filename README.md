
# Solar Power Energy Prediction Project

This project predicts solar power energy generation using a trained ML model with a Flask web interface.

---

## Project Structure

```
Solar Project ExcelR/
│
├── app.py                      # Main Flask app
├── requirements.txt            # Python dependencies
├── README.md                   # Project instructions
│
├── data/                       # Store all CSV/raw data files
│   ├── solarpowergeneration.csv
│   └── solarpowergeneration_cleaned.csv
│
├── models/                     # Trained ML models and CatBoost info
│   └── my_trained_models.pkl
│
├── notebooks/                  # Jupyter notebooks for exploration and modeling
│   ├── analysis.ipynb          # First notebook to execute
│   ├── preprocessing.ipynb     # Second notebook to execute
│   └── modeling.ipynb          # Third notebook to execute
│
├── static/                     # CSS for Flask
│   └── style.css
│
├── templates/                  # HTML templates for Flask
│   └── index.html

```

## Setup & Execution (Windows - CMD)

1. Open Command Prompt and navigate to project folder:
   Ex:- cd "D:\Data Science Excelr\Solar Project ExcelR"

2. Create Virtual Environment for Once:
   python -m venv venv

3. Activate the virtual environment:
   venv\Scripts\activate

4. Upgrade pip:
   python -m pip install --upgrade pip

5. Install project dependencies:
   pip install -r requirements.txt

6. Run the Flask app:
   python app.py

7. Open your browser and go to the link shown (usually (http://127.0.0.1:5000)):
   Press CTRL+C to quit.

8. Deactivate the virtual environment:
   deactivate



# Solar Power Energy Prediction Project - Comprehensive Pipeline

## Phase 1: Data Analysis — Solar Power Energy Prediction

### 1.1 Initial Data Exploration

#### 1.1.1 Environment Setup
- Import core libraries: `pandas`, `numpy` for data manipulation.
- Import visualization libraries: `matplotlib.pyplot`, `seaborn`, `statsmodels`.
- Suppress warnings and configure Pandas display options.

#### 1.1.2 Data Loading
- Define file path configuration and load dataset using `pd.read_csv()`.
- Implement robust error handling for file loading.

#### 1.1.3 Basic Data Inspection
- Display first 5 rows of the dataset using `.head()`.
- Check the shape of the dataset with `data.shape` (2920 rows × 10 columns).
- List all column names: `data.columns`.
- Count unique values per column to analyze cardinality.

#### 1.1.4 Data Type Analysis
- Verify data types of columns using `data.dtypes`.
- Confirm that there are no object/string columns. Identify numeric columns (float64, int64).

#### 1.1.5 Data Quality Assessment
- Check for duplicate rows.
- Verify consistency with non-null counts.
- Generate a statistical summary using `.describe()` for mean, std, min, max, etc.

### 1.2 Missing Value Analysis

#### 1.2.1 Identify Missing Values
- Check for missing values with `data.isna().sum()`.
- Identify columns with missing values and calculate the percentage of missing data.

#### 1.2.2 Missing Value Decision Strategy
- Drop the single missing value in `average-wind-speed-(period)` using `data.dropna(inplace=True)`.

### 1.3 Outlier Detection and Analysis

#### 1.3.1 Statistical Definition
- Define outliers using the Interquartile Range (IQR) method: thresholds are `Q1 - 1.5 * IQR` and `Q3 + 1.5 * IQR`.

#### 1.3.2 Function Implementation
- Develop functions `outlier()`, `check_outlier()`, and `detect_outliers()` to detect and handle outliers.

#### 1.3.3 Systematic Column Scan
- `wind-direction`: 18% outliers (artifact of scale).
- `visibility`: 14.6% outliers.
- `power-generated`: 3.6% outliers.

#### 1.3.4 Contextual Logic & Domain Analysis
- Removed 13 rows where `distance < 0.5` (Noon) and `sky-cover == 0` (Clear) but `power == 0` (Sensor failure).

### 1.5 Exploratory Data Analysis (EDA)

#### 1.5.1 Histogram Visualization
- Plotted histograms for all feature columns.
- Observations: Temperature and pressure are approximately normal, wind-related variables are skewed.

#### 1.5.2 Boxplot Visualization
- Generated boxplots to detect outliers and variability.
- Wind-related variables, visibility, and humidity show visible outliers, indicating rare weather events.

#### 1.5.3 Correlation Analysis
- Computed Pearson correlation matrix and plotted heatmap.
- Observations: Strong negative correlation between `power-generated` and `distance-to-solar-noon` and `humidity`. Mild multicollinearity observed between `wind speed` and `average wind speed`.

#### 1.5.4 Feature vs Target Relationship (Scatter Plots)
- Scatter plots of each feature against `power-generated`.
- Observations: Power generation peaks near solar noon and under clear skies (low humidity/sky-cover).

#### 1.5.5 Multicollinearity Check 
- Severe multicollinearity detected among `temperature`, `pressure`, `visibility`, and `humidity`.

#### 1.5.6 Q-Q Plots & Normality Checks
- Normality tests on temperature, average pressure, and wind speed.
- Power generation is zero-inflated and non-normal.

#### 1.5.7 Target-Based Binning Analysis
- Binned `power-generated` into categories (Low, Medium, High) and compared features.
- High power corresponds to low distance to solar noon, lower humidity, and slightly higher temperature.

#### 1.5.9 PCA for Feature Redundancy
- Applied PCA and explained 90% variance with the top 5 components.

#### 1.5.10 Baseline Linear Model Diagnostics
- Fitted OLS model and checked for Cook's Distance, confirming no dominant observations.

### Outcome

The data analysis phase provides a comprehensive understanding of the dataset with:  
- Identification of missing values and minimal cleaning applied  
- Detection of statistical and contextual outliers  
- Understanding of feature distributions, correlations, and multicollinearity  
- Insights on feature-target relationships and zero-inflated target behavior  
- PCA applied for feature redundancy assessment  
- Baseline linear model diagnostics performed to check data suitability for modeling


## Phase 2: Data Preprocessing — Solar Power Energy Prediction

This phase prepares the raw dataset for modeling by cleaning erroneous values, handling outliers, engineering features, reducing redundancy, and exporting a finalized dataset for model training.

---

### Prerequisite
The analysis phase must be completed before running this step.

Required file:
- `analysis.ipynb`
---

### 2.1 Outlier Treatment & Feature Engineering

Based on insights from the analysis phase, several data quality issues and domain-specific inconsistencies were addressed.

---

#### 2.1.1 Data Loading
- Dataset loaded from `solarpowergeneration.csv`
- Same source file as used in the analysis phase

---

#### 2.1.2 Missing Value Removal
- Identified a small number of missing values
- Rows containing missing values were dropped
- Dataset verified to contain no remaining null values

---

#### 2.1.3 Removal of Physically Impossible Values
Rule-based filtering was applied to remove invalid measurements:
- wind-speed ≥ 0  
- average-pressure-(period) ≥ 0  
- humidity ≥ 0  
- average-wind-speed-(period) ≥ 0  

These conditions enforce basic physical constraints.

---

#### 2.1.4 Contextual Outlier Handling (Domain Logic)

##### Night-Time Generation Check
- Observations with large distance-to-solar-noon but non-zero power generation were flagged as suspicious

##### Day-Time Zero Generation Check
- Rows with:
  - distance-to-solar-noon < 0.5
  - sky-cover = 0
  - power-generated = 0  
- Interpreted as sensor failures and removed

Affected rows were dropped and the index was reset.

---

#### 2.1.5 Cyclic Encoding of Wind Direction
- Original wind-direction (1–36 scale) converted to degrees
- Cyclic encoding applied using sine and cosine transformations
- Original wind-direction column removed

This preserves directional continuity and avoids artificial ordinal relationships.

---

### 2.2 Multicollinearity Check & Feature Selection

#### 2.2.1 Variance Inflation Factor (VIF)
- VIF computed for all features
- No severe multicollinearity detected
- VIF used only to assess linear redundancy, not feature usefulness

---

#### 2.2.2 Cross-Validated Feature Ablation
- Performed drop-feature ablation using cross-validated MAE
- GradientBoostingRegressor used as the evaluation model

**Findings:**
- Strong predictors:
  - distance-to-solar-noon
  - sky-cover
  - humidity
- Moderate contributors:
  - temperature
  - wind-speed
  - average-wind-speed-(period)
  - wind direction components
- Low or negligible impact:
  - visibility
  - average-pressure-(period)

---

#### 2.2.3 Correlation-Based Redundancy Check
- Pairwise correlation analysis performed
- Highly correlated feature pairs identified
- Redundant features reviewed in combination with ablation results

---

#### 2.2.4 Feature Removal
- `visibility` removed due to low predictive contribution

---

### 2.3 Saving the Cleaned Dataset

#### 2.3.1 Export
- Cleaned dataset saved as:
data/solarpowergeneration_cleaned.csv

- File creation verified successfully

---

#### 2.3.2 Documented Preprocessing Steps
Applied transformations include:
1. Removal of missing values  
2. Statistical and domain-based outlier handling  
3. Cyclic encoding of wind direction  
4. Multicollinearity analysis and feature selection  

---

### 2.4 Reloading Cleaned Data
- Cleaned dataset reloaded from CSV
- Shape and structure verified

---

### 2.5 Train–Test Split
- Features and target separated
- 80/20 train–test split applied
- Random seed fixed for reproducibility

---

### 2.6 Feature Scaling & Target Transformation

#### Feature Scaling
- **RobustScaler**
  - wind-speed
  - humidity
  - average-wind-speed-(period)

- **StandardScaler**
  - distance-to-solar-noon
  - temperature

- Remaining features passed through unchanged
- Scaling fit only on training data using a ColumnTransformer

---

#### Target Transformation
- Target variable `power-generated` log-transformed using `log1p()`
- Transformation reduces skewness and stabilizes variance
- Predictions are inverse-transformed using `expm1()`

---

### Outcome

The preprocessing phase produces a clean, consistent, and model-ready dataset with:
- Physically valid observations
- Reduced noise and redundancy
- Properly encoded cyclic features
- Scaled inputs and stabilized target variable

This dataset is used directly in **Phase 3: Modeling**.

## Phase 3: Modeling — Solar Power Energy Prediction

This phase focuses on training, tuning, evaluating, and saving machine learning models to predict solar power generation using the cleaned and feature-engineered dataset produced in the preprocessing phase.

---

### Prerequisite
The preprocessing step must be completed before running this phase.

Required file:
- preprocessing.ipynb

---

### 3.1 Data Preparation

#### 3.1.1 Feature and Target Definition
- Target variable: `power-generated`
- Feature set: all remaining engineered variables
- Dataset split into training and testing sets using an 80/20 split

---

#### 3.1.2 Feature Scaling
Different scalers are applied based on feature distributions:

- RobustScaler
  - wind-speed
  - humidity
  - average-wind-speed-(period)

- StandardScaler
  - distance-to-solar-noon
  - temperature

A ColumnTransformer is used to apply scaling while passing remaining features unchanged.  
Scalers are fit only on the training data.

---

#### 3.1.3 Target Transformation
- The target variable is skewed and zero-inflated
- Logarithmic transformation applied:
- Predictions are inverse-transformed using expm1() during evaluation.

### 3.3 Model Metric Evaluation

**Performance Summary:**

- **Decision Tree:**  
  - Captures non-linear patterns but tends to overfit and generalizes poorly.

- **Random Forest:**  
  - Reduces variance, improves generalization, and performs better than a single tree.

- **Boosting Models:**  
  - XGBoost: Test R² ≈ 0.89, lowest MAE and RMSE  
  - LightGBM: Slightly lower training error, Test R² ≈ 0.89  
  - Capture complex non-linear patterns and generalize well  
  - Overall best-performing models
  - Capture complex non-linear relationships and generalize well.  
  - Overall best-performing models.

**Conclusion:** Among all models, **XGBoost** is selected as the primary model for deployment due to its best balance of training fit, test performance, and generalization ability.

---

### 3.4 Model Persistence

**Saved Models:**  
- Decision Tree  
- Random Forest  
- XGBoost  
- LightGBM  
- CatBoost  

**File:** `models/my_trained_models.pkl`

### Outcome

The modeling phase produces trained and ready-to-use predictive models with:  
- Decision Tree and Random Forest capturing non-linear patterns  
- Boosting models (XGBoost, LightGBM, CatBoost) capturing complex relationships  
- XGBoost selected as primary model for deployment due to best generalization  
- All models saved for direct prediction without retraining


## Phase 4: Model Deployment — Solar Power Energy Prediction

The trained **XGBoost model** is deployed as a web application using **Flask**. Users can input feature values through a web interface to get real-time solar power predictions.

### 4.1 Web Application Structure

- **app.py**: Main Flask application that handles:
  - Loading trained models from `models/my_trained_models.pkl`
  - Accepting user input through the web form
  - Preprocessing input data to match model requirements
  - Returning predicted power values to the user

- **templates/index.html**: HTML template for the web interface:
  - Input fields for all features used in the model
  - Submit button to trigger prediction
  - Output area to display predicted solar power

- **static/style.css**: CSS file for styling the web application:
  - Layout, colors, fonts, and responsive design

### 4.2 Usage Instructions

1. Open Command Prompt and navigate to project folder:
   Ex:- cd "D:\Data Science Excelr\Solar Project ExcelR"

2. Create Virtual Environment for Once:
   python -m venv venv

3. Activate the virtual environment:
   venv\Scripts\activate

4. Upgrade pip:
   python -m pip install --upgrade pip

5. Install project dependencies:
   pip install -r requirements.txt

6. Run the Flask app:
   python app.py

7. Open your browser and go to the link shown (usually (http://127.0.0.1:5000)):
   Press CTRL+C to quit.

8. Deactivate the virtual environment:
   deactivate

### Outcome

The deployment phase provides a fully functional web application with:

- Real-time solar power predictions using the trained **XGBoost model**
- User-friendly input interface via `index.html`
- Styled and responsive design using `style.css`
- Flask backend in `app.py` that handles preprocessing, prediction, and result display
- Reusable trained models stored in `models/my_trained_models.pkl`
- End-to-end integration of data preprocessing, modeling, and deployment


