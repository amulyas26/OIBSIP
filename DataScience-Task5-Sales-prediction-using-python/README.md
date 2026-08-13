TASK 5- SALES PREDICTION USING PYTHON

Problem Statement: Explains what sales prediction is and why companies use machine learning to analyze marketing budgets.

Dataset Context: Introduces advertising.csv and its core feature columns (TV, Radio, Newspaper, Sales).

End-to-End Objective: Clear bullet points detailing the entire pipeline you implemented—from EDA and data splitting to model comparison, residual analysis, and budget forecasting.


📁 1. Explanation of advertising.csv

The advertising.csv dataset is a classic marketing analytics file used to analyze the relationship between advertising spending across different channels and product sales revenue.

File Structure & Columns:
TV (Numeric): The total budget spent on TV advertisements for a product (in thousands of dollars).
Radio (Numeric): The total budget spent on Radio advertisements (in thousands of dollars).
Newspaper (Numeric): The total budget spent on Newspaper advertisements (in thousands of dollars).
Sales (Numeric / Target Variable): The target metric representing product sales units (in thousands of units) generated as a result of the advertising spend.
Unnamed: 0 (Index Column): An optional raw index column from the original CSV export, which is safely dropped during preprocessing.

📓 2. Explanation of sales_prediction.ipynb 

Below is an explanation of every step inside your notebook, detailing what the Markdown cell says, how the code functions under the hood, and what output is rendered in the notebook.

Step 1: Imports & Data Loading
Code Explanation:
Imports key libraries: pandas and numpy for data manipulation, matplotlib and seaborn for plotting, and scikit-learn submodules for model building (LinearRegression, RandomForestRegressor) and metrics (mean_absolute_error, mean_squared_error, r2_score). It then loads advertising.csv into a DataFrame and drops the Unnamed: 0 index column if it exists.
Rendered Output:
A Pandas DataFrame table previewing the first 5 rows (df.head()) showing numeric columns for TV, Radio, Newspaper, and Sales.
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/5ce7a21f-a7fc-4a76-a7ad-00acf5318db6" />

Step 2: Exploratory Data Analysis (EDA)
Code Explanation:
Runs structural sanity checks on the dataset. It prints column data types, verifies zero null values via .isnull().sum(), and prints summary statistics (mean, min, max, standard deviation). It then generates a pairwise relationship matrix (pairplot) and a correlation heatmap matrix to visualize linear correlations between features and sales.
Rendered Output:
Printed dataset summary showing 200 non-null rows per feature.
A 4 \times 4 Pairplot grid showing scatter graphs between all column pairs.
A Correlation Heatmap revealing high correlation between TV and Sales (~0.90), moderate correlation for Radio (~0.35), and low correlation for Newspaper (~0.16).
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/4a73169a-53fa-4136-82a3-547564a536ba" />
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/752ab952-f005-4e6c-831d-f3cee3982aca" />
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/bd77cbe7-920a-4fdf-bb93-fc3fe5732ddc" />

Step 3: Data Splitting (Train/Test Split)
Code Explanation:
Isolates feature variables (X = \text{TV, Radio, Newspaper}) from the target variable (y = \text{Sales}). It calls train_test_split() using an 80/20 ratio and random_state=42 to ensure consistent data partition for reproducible evaluation.
Rendered Output:
Printed sample count confirmation:
Training sample count: 160
Testing sample count : 40
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/78208a43-bc45-4558-87c9-c14fe80e1191" />

Step 4: Train Models
Markdown Header: # 4. Train Models
Code Explanation:
Instantiates and fits two regression algorithms using X_train and y_train:
Linear Regression (lr): Learns linear weight coefficients for each media channel.
Random Forest Regressor (rf): Builds an ensemble of decision trees to capture non-linear relationships and feature interactions.
Both models then output prediction vectors (lr_preds and rf_preds) on X_test.
Rendered Output:
Notebook cell execution status tick mark showing execution time (~1.5s - 20s depending on hardware).
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/56dd89f2-9b99-4b07-82d5-e50c9ec2e444" />

Step 5: Model Evaluation
Code Explanation:
Defines a custom helper function evaluate_metrics() that calculates three performance metrics on the test predictions:
MAE (Mean Absolute Error): Average unit deviation from actual sales.
RMSE (Root Mean Squared Error): Penalizes larger prediction outliers.
R^2 Score: Percentage of target variance explained by the model.
Rendered Output:
=== Linear Regression ===
MAE:  1.2748
RMSE: 1.7052
R²:   0.9059

=== Random Forest Regressor ===
MAE:  0.9180
RMSE: 1.1989
R²:   0.9535
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/681eddc6-01e1-4db3-84ee-7aea3e6d70cb" />

Step 6: Residual Plot & Channel Impact Analysis
Code Explanation:
Calculates residuals (\text{Actual} - \text{Predicted}) for Linear Regression and generates a scatter plot against predicted values with a horizontal dashed reference line at zero. It then extracts lr.coef_ into a structured Pandas DataFrame to analyze the exact linear weight of each media channel.
Rendered Output:
Residual Plot: A scatter plot showing residuals randomly distributed around the red zero line without funneling, confirming linear regression assumptions hold.
Coefficients Table:
TV: ~0.0545 (Highest return per dollar spent)
Radio: ~0.1009
Newspaper: ~0.0067 (Minimal impact on final sales)
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/90a660cc-ce69-4702-b60c-071c6ba33b6f" />

Step 7: Random Forest Feature Importance
Markdown Header: ### Random Forest Feature Importance
Explanation:
While Linear Regression uses fixed coefficients to show channel impact, the Random Forest model calculates Gini Importance (or Mean Decrease in Impurity). It evaluates how much each advertising channel reduces variance across all decision trees in the forest.
Output & Finding:
A horizontal bar chart displaying relative importance scores.
TV advertising dominates with an importance score of around ~80–85%, making it the primary driver of sales in the ensemble model.
Radio accounts for around ~12–15%, while Newspaper contributes less than ~3%, confirming that Newspaper spend adds very little predictive value.
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/86a02bbe-c2ee-4d6a-8049-e849e7c781ab" />

Step 8: Actual vs. Predicted Sales Comparison
Explanation:
This visual comparison plots true target values (y_test) on the X-axis against model predictions (preds) on the Y-axis. A red dashed diagonal line (y = x) represents perfect prediction accuracy (where predicted sales exactly match real sales).
Output & Finding:
Random Forest points (dark green) form a tight cluster directly along the red diagonal reference line, reflecting its high R^2 score (0.9535).
Linear Regression points (orange) show wider dispersion around the line, visually proving why the non-linear Random Forest model outperforms the baseline.
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/72cd89d2-df13-4211-8978-c8ae854ddbe5" />

Step 9: Sales Prediction Function (predict_sales)
Explanation:
This is a practical business utility function that turns your trained machine learning model into an interactive tool. It accepts custom dollar inputs for TV, Radio, and Newspaper spends, formats them into a single-row DataFrame, and uses rf.predict() to calculate estimated sales units
Output & Scenario Example:
--- Budget Scenario ---
TV Spend       : $200.00
Radio Spend    : $40.00
Newspaper Spend: $10.00
-----------------------
Estimated Sales: 19.82 units
This demonstrates real-world application, allowing marketing teams to simulate different budget allocations before spending actual capital.
<img width="399" height="244" alt="image" src="https://github.com/user-attachments/assets/72a0167e-073a-4e6a-8775-abdc9f377694" />


 Conclusion & Final Executive Summary

📌 Project Overview
In this task, we built an end-to-end Machine Learning pipeline to predict product sales based on promotional expenditures across **TV**, **Radio**, and **Newspaper** advertising channels using the `advertising.csv` dataset. The goal was to quantify channel impact, compare predictive algorithms, and provide actionable business insights for marketing budget optimization.

---

 📊 Key Findings & Model Comparison

1. **Model Performance:**
   * **Random Forest Regressor** proved to be the superior model, achieving an **$R^2$ score of 0.9535** ($\text{MAE} = 0.9180$, $\text{RMSE} = 1.1989$).
   * The baseline **Linear Regression** model performed strongly with an **$R^2$ score of 0.9059** ($\text{MAE} = 1.2748$, $\text{RMSE} = 1.7052$), but was unable to capture subtle non-linear interactions between channels as effectively as the ensemble model.

2. **Channel Impact & ROI Analysis:**
   * **TV Advertising:** Identified as the primary driver of sales revenue across both linear coefficients and Random Forest feature importances (~80–85% importance score). Increasing TV ad spend yields the highest return on investment.
   * **Radio Advertising:** Showed a solid positive impact (~12–15% importance score), acting as an effective secondary support channel.
   * **Newspaper Advertising:** Demonstrated minimal to negligible impact on total sales (<3% importance score), suggesting that capital allocated here yields diminishing returns.

3. **Model Validation & Reliability:**
   * Residual analysis confirmed that prediction errors are randomly scattered around zero without systematic heteroscedasticity or funneling patterns.
   * Actual vs. Predicted plots verified that Random Forest predictions tightly follow the ideal diagonal fit line, confirming high model reliability for decision-making.


 💡 Strategic Business Recommendations

* **Reallocate Budgets:** Shift low-performing advertising budgets from **Newspaper** into **TV** to maximize sales volume.
* **Combine Channels:** Utilize **Radio** in tandem with **TV** campaigns to capture cross-channel synergy.
* **Scenario Planning:** Use the implemented `predict_sales()` utility to simulate hypothetical budget distributions before committing real marketing capital.
