🚗 Car Price Prediction Engine
An end-to-end Machine Learning web application estimating pre-owned vehicle resale values using historical market data, Python, scikit-learn, and Gradio.

📁 Repos
itory Structure
📂 OIBSIP
├── 📊 car data.csv                 # Historical sales dataset (301 records)
├── 📓 car_price_prediction.ipynb   # Complete ML pipeline & embedded UI
└── 📄 README.md                    # Project documentation


1. 📊 car data.csv (The Raw Dataset)
This dataset contains historical secondary market transaction records used to train and test our predictive model.
 * 🏷️ Car_Name: Brand and model name of the vehicle (e.g., Maruti Swift, Honda City, Hyundai i20).
 * 📅 Year: Manufacturing/purchase year of the vehicle.
 * 💰 Selling_Price: Target Variable (y) — The actual price the car was sold for (in Lakhs ₹).
 * 🏷️ Present_Price: Current ex-showroom price of the vehicle when new (in Lakhs ₹).
 * 🛣️ Kms_Driven: Total cumulative distance traveled by the vehicle in kilometers.
 * ⛽ Fuel_Type: Fuel mechanism (Petrol, Diesel, or CNG).
 * 🤝 Seller_Type: Entity selling the car (Dealer vs. Individual).
 * ⚙️ Transmission: Gearbox configuration (Manual vs. Automatic).
 * 👤 Owner: Number of previous registered owners (0, 1, 2, or 3).

2. 📓 car_price_prediction.ipynb (The Intelligence Core)
This Jupyter Notebook houses the complete machine learning pipeline:
 * 📦 Data Ingestion & Library Setup: Loads fundamental libraries (pandas, numpy, matplotlib, seaborn) and parses car data.csv.
 * 🧹 Cleaning & Feature Engineering: Validates missing data, computes vehicle age (current_year - df['Year']), and drops uninformative text identifiers.
 * 🔢 Categorical Encoding: Applies one-hot encoding via pd.get_dummies() to convert text labels into binary matrices (0 or 1) for mathematical model compatibility.
 * ✂️ Dataset Partitioning: Splits the clean data into predictor features (X) and targets (y), reserving 80% for training and 20% for testing.
 * 🤖 Model Fitting & Evaluation: Trains a regression model (e.g., Random Forest / Linear Regression) to learn price depreciation trends and measures prediction accuracy.
 * 🎨 Interactive Interface (Gradio): Launches a front-end UI directly inside the notebook cell, allowing users to tweak vehicle specs via sliders and radio buttons to calculate immediate price estimates.
* 📊 Exploratory Data Analysis (EDA): Visualizes price distributions, categorical feature trends, and numerical feature correlation heatmaps to extract key dataset insights.
* ⚔️ Multi-Model Benchmarking: Trains and compares multiple algorithms (Random Forest Regressor vs. Linear Regression) using standard regression metrics ($R^2$, MAE, RMSE).
* 💡 Feature Importance Analysis: Plots relative feature contribution scores to identify primary valuation drivers (e.g., Present Price, Vehicle Age).
* 🗣️ User Input Prediction Utility: Provides a function to run real-time inference on custom vehicle parameters to predict resale market value.


Step-by-step process of entire Car_price_prediction task

1. Data Ingestion & Setup
Code: Imports pandas, numpy, matplotlib, seaborn, and datetime. Reads raw data via df = pd.read_csv('car data.csv') and inspects initial rows with df.head().
Output: Displays original vehicle attributes including brand, purchase year, target Selling_Price, Present_Price, Kms_Driven, and categorical features.
<img width="1600" height="294" alt="image" src="https://github.com/user-attachments/assets/dc7e2343-7ed3-4d4e-99c0-09cf5c75a104" />

2. Data Cleaning & Feature Engineering
Code: Validates dataset integrity using df.isnull().sum(). Computes vehicle age via Car_Age = current_year - df['Year'], drops redundant columns (Car_Name, Year), and encodes categorical variables into binary columns using pd.get_dummies(..., drop_first=True).
Output: Confirms zero missing values and outputs a clean, all-numeric DataFrame ready for modeling.
<img width="1280" height="254" alt="image" src="https://github.com/user-attachments/assets/8b162866-0a5b-4b8a-b85e-d542fcaa762c" />

3. Model Training & Evaluation
Code: Separates predictors (X) from the target (y), splits data using train_test_split() (80\% train, 20\% test), fits a RandomForestRegressor(), and evaluates performance using R^2 score and MAE metrics.
Output: Achieves an R^2 Score of 0.9607 (96.07\% price variance explained) and a Mean Absolute Error of ₹0.6326 Lakhs (~₹63,256 average deviation).
<img width="1600" height="279" alt="image" src="https://github.com/user-attachments/assets/d60ebf58-d5df-48ef-9a5f-72d627454887" />

4. Model Performance & Residual Analysis
Code: Generates an Actual vs. Predicted scatter plot (plt.scatter) with a diagonal reference line and plots Random Forest feature importances (model.feature_importances_).
Output: Shows predicted prices closely tracking actual values along the reference line. Identifies Present_Price as the strongest valuation factor, followed by Car_Age and Kms_Driven.
<img width="720" height="571" alt="image" src="https://github.com/user-attachments/assets/81021504-29f4-4da1-9ab6-0a35c015e66a" />

5. Single-Sample Inference & Model Verification
Code: Extracts a single test record using X_test.iloc[0:1] and runs model.predict() to test individual price generation.
Output: Displays Actual Price: ₹0.35 Lakhs vs. Predicted Price: ₹0.44 Lakhs, validating instant inference capability for real-world deployment.
<img width="510" height="103" alt="image" src="https://github.com/user-attachments/assets/eb4e32a5-0dc6-44a1-94e9-9de1a2e0d931" />

6. Data Ingestion & Environment Setup
Code: Imports necessary libraries including datetime, joblib, matplotlib.pyplot, numpy, pandas, seaborn, sklearn estimators/metrics, and gradio. Reads car data.csv into a Pandas DataFrame and calls display(df.head()).
Output: Displays the top rows of the raw dataset with features like Car_Name, Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, and Owner.
<img width="1280" height="252" alt="image" src="https://github.com/user-attachments/assets/0d59948c-7a7d-4159-90c9-333b1468fe98" />

7. Dataset Partitioning & Feature Selection
Code: Separates independent predictor features (X = df.drop('Selling_Price', axis=1)) from the target variable (y = df['Selling_Price']). Splits the data into 80% training and 20% testing sets via train_test_split(X, y, test_size=0.2, random_state=42).
Output: Prepares X_train, X_test, y_train, and y_test arrays ready for model training.

8. Data Loading & Preprocessing Pipeline
Code: Demonstrates a clean, end-to-end preprocessing setup: loads car data.csv, derives Car_Age = 2026 - df['Year'], drops redundant features (Car_Name, Year), one-hot encodes categorical attributes (pd.get_dummies(..., drop_first=True)), and separates predictors (X) from the target (y).
Output: Prepares the fully numeric feature matrix X and target vector y for streamlined pipeline execution.
<img width="989" height="559" alt="image" src="https://github.com/user-attachments/assets/2f97d194-b275-4f71-8f97-42f4aeb61c8a" />

9. Initial Exploratory Data Analysis & Feature Importance
Code: Calculates and plots feature importances using model.feature_importances_ in a horizontal bar chart (kind='barh'). Uses seaborn to generate a 3-panel visualization: a histogram of Selling_Price distribution (sns.histplot), a boxplot comparing price across Fuel_Type (sns.boxplot), and a scatter plot comparing price against Car_Age (sns.scatterplot).
Output:
Feature Importance Chart: Highlights Present_Price as the most critical factor, followed by Car_Age and Kms_Driven.
EDA Plots: Shows right-skewed selling prices, higher median prices for diesel vehicles, and a clear downward price trend as vehicle age increases.
<img width="1280" height="338" alt="image" src="https://github.com/user-attachments/assets/ded1337e-97e3-470a-9590-f6e382eb5fb8" />

10. Feature Correlation Heatmap
Code: Filters numerical columns (df.select_dtypes(include=['number'])), calculates pairwise Pearson correlation coefficients (numeric_df.corr()), and renders a color-coded heatmap using sns.heatmap(..., annot=True, cmap='coolwarm').
Output: Displays a correlation matrix highlighting strong positive linear correlation (0.88) between Present_Price and Selling_Price.
<img width="1280" height="682" alt="image" src="https://github.com/user-attachments/assets/6bb02806-331e-4b12-88fb-b85a8f6292c2" />

11. Linear Regression & Model Comparison
Code: Trains a baseline LinearRegression model (lr_model.fit(X_train, y_train)), generates predictions for both models (lr_preds and rf_preds), and prints comparative performance metrics (R^2, MAE, and RMSE).
Output: Prints formatted metric comparisons under === Model Comparison === to contrast the predictive performance of Random Forest against Linear Regression on the test dataset.
<img width="778" height="123" alt="image" src="https://github.com/user-attachments/assets/34dc0a91-08fe-4c9c-b883-c9f66c791788" />


Bullet Points (Best for a Features / Tech Stack Section)

⚡ Gradio Integration: Powered by Gradio, an open-source Python framework designed for rapid prototyping and interactive Machine Learning web applications.
🎛️ Real-Time UI Components: Converts complex backend model predictions into simple, accessible visual controls such as sliders, dropdowns, and radio buttons.
🌐 Instant Web Deployment: Generates an embedded interface within Jupyter Notebooks alongside a public link (.gradio.live) for effortless sharing and evaluation.


🎯 Conclusion

The Car Price Prediction Engine successfully demonstrates a complete, end-to-end Machine Learning workflow—bridging raw historical data with a functional user deployment:
📊 Data-Driven Insights (car data.csv): By analyzing secondary vehicle transaction data, key pricing drivers such as original showroom cost, cumulative mileage, vehicle age, and fuel/gearbox configurations were isolated and quantified.* 📊 Exploratory Data Analysis (EDA): Visualizes price distributions, categorical feature trends, and numerical feature correlation heatmaps to extract key dataset insights.
* ⚔️ Multi-Model Benchmarking: Trains and compares multiple algorithms (Random Forest Regressor vs. Linear Regression) using standard regression metrics ($R^2$, MAE, RMSE).
* 💡 Feature Importance Analysis: Plots relative feature contribution scores to identify primary valuation drivers (e.g., Present Price, Vehicle Age).
* 🗣️ User Input Prediction Utility: Provides a function to run real-time inference on custom vehicle parameters to predict resale market value.

🤖 Robust Predictive Modeling (car_price_prediction.ipynb): Through rigorous preprocessing, feature engineering, and categorical encoding, the notebook transforms unstructured attributes into structured feature vectors. Fitting an ensemble regressor allows the system to accurately map complex non-linear depreciation curves.Through systematic preprocessing, feature engineering, and categorical encoding, the dataset was evaluated across multiple algorithms (Random Forest Regressor vs. Linear Regression). The ensemble approach captured complex non-linear depreciation curves and outperformed linear baselines across key evaluation metrics ($R^2$, MAE, RMSE).

🌐 Seamless Deployment (Gradio Interface): Wrapping the trained model in an interactive, user-friendly frontend turns complex algorithmic predictions into an intuitive tool—allowing users to receive instant, real-time market valuations with simple slider adjustments.Wrapping the trained Random Forest model in an interactive, user-friendly frontend turns complex algorithmic predictions into an intuitive tool—allowing users to receive instant, real-time market valuations with simple slider adjustments.
