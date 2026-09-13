# 🏠 House Price Prediction in California

## Project Overview

This project develops a machine learning model to predict house prices based on property, location, construction, and sale-related features.

A **Random Forest Regression** model was selected to learn the relationship between these features and house prices. The project covers the complete machine learning workflow, including data analysis, preprocessing, model training, parameter evaluation, cross-validation, error analysis, feature importance analysis, and deployment.

The final model was also integrated into a **Streamlit web application**, allowing users to enter house information and receive an estimated house price.

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze a real-world house price dataset.
- Identify the factors that influence house prices.
- Develop a regression model for house price prediction.
- Evaluate the model using appropriate regression metrics.
- Analyze prediction errors and model limitations.
- Deploy the trained model through a user-friendly web application.

---

## 📊 Dataset

The dataset contains **14,620 house records** and **22 input features**.

The features include:

- Number of bedrooms
- Number of bathrooms
- Living area
- Lot area
- Number of floors
- Waterfront presence
- Number of views
- House condition
- House grade
- House area excluding basement
- Basement area
- Built year
- Renovation year
- Postal code
- Latitude
- Longitude
- Renovated living area
- Renovated lot area
- Number of schools nearby
- Distance from the airport
- Sale year
- Sale month

The target variable is:

```text
Price

🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Jupyter Notebook
Joblib
Streamlit
🤖 Machine Learning Model

The final model uses:

Random Forest Regression

Final configuration:

RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

Several Random Forest parameters were investigated during model development, including:

Maximum depth
Minimum samples per leaf
Minimum samples for splitting
Maximum features
Number of estimators

Additional experiments involving feature engineering and target transformation were also performed. The original Random Forest model provided the strongest overall result and was selected as the final model.

📈 Model Performance

The final model was evaluated using a separate testing dataset.

Metric	Result
R² Score	0.8747 (87.47%)
MAE	68,636.02
MSE	17,660,748,550.08
RMSE	132,893.75
Cross-Validation

Five-fold cross-validation produced:

Average R²: 0.9351
Standard Deviation: 0.0038

This indicates consistent model performance across the validation folds.

🔍 Feature Importance

The most important features identified by the Random Forest model were:

Feature	Importance
Living area	0.346629
Grade of the house	0.232991
Latitude	0.151355
Longitude	0.064222
Renovated living area	0.032731
House area excluding basement	0.031092
Waterfront presence	0.029990
Built year	0.020181
Postal code	0.015709
Number of views	0.013062

The results show that living area and house grade have the strongest influence on the model's predictions among the analyzed features.

📉 Error Analysis

Residual and prediction error analysis showed that most predictions were relatively close to the actual house prices.

However, prediction errors increased for some higher-priced properties. The dataset also contains a small number of very high-priced properties, which makes prediction more difficult for these cases.

The residual analysis showed that the errors were generally distributed around zero, while larger errors appeared more frequently at higher predicted prices.

🌐 Streamlit Application

The trained model was integrated into a Streamlit web application.

The application allows users to enter property information such as:

Property size
Bedrooms and bathrooms
House condition and grade
Construction information
Location
Waterfront and views
Nearby schools
Sale date

After entering the information, the application uses the trained Random Forest model to generate an estimated house price.

📁 Project Structure
house-price-prediction-california/
│
├── app.py
├── requirements.txt
├── house_price_prediction.ipynb
├── .gitignore
└── README.md

The trained model file is not included in the repository because its size exceeds GitHub's individual file-size limit.

▶️ How to Run the Project
1. Clone the repository
git clone https://github.com/GHODI132/house-price-prediction-california.git
2. Open the project folder
cd house-price-prediction-california
3. Install the required libraries
pip install -r requirements.txt
4. Run the Streamlit application
streamlit run app.py
⚠️ Limitations

The model does not predict every house price with the same level of accuracy.

The analysis showed larger prediction errors for some high-priced and unusual properties. This is partly related to the strong right-skewness of the price distribution and the limited number of extremely expensive houses in the dataset.

The predicted price should therefore be treated as an estimate rather than a guaranteed market price.

🚀 Future Improvements

Possible future improvements include:

Testing additional regression algorithms.
Using larger and more diverse datasets.
Adding more property and location features.
Exploring advanced hyperparameter optimization.
Improving the handling of high-priced outliers.
Deploying the application on a public cloud platform.
👨‍💻 Author

Aghiad Al Jbaaee

B.Tech CSE (AIML)
Geeta University, Panipat


### A small but important point

I deliberately **didn't claim that the project uses California-specific data beyond what we have established in the project title**. The actual feature list and results are based on our work, and the README should remain technically honest.

Also, notice that we don't say:

> "87.47% accuracy"

We say:

> **R² Score = 0.8747 (87.47%)**

That's the technically correct terminology for our regression model.

---

## Step 8.1 — Add the README to Git

After saving `README.md`, run:

```bash
git status