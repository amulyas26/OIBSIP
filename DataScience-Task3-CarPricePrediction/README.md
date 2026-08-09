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