Spam Detection System

An end-to-end Machine Learning project that classifies raw text messages as SPAM or HAM (legitimate). The repository contains the complete dataset, model training workflow, serialized inference artifacts, and generated evaluation reports.


Core Files Overview

*spam.csv: Raw dataset containing 5,572 tagged SMS/email messages used for model training and testing (86.59% Ham, 13.41% Spam).   
*spam_detection.ipynb: Interactive Jupyter Notebook performing EDA, text preprocessing (cleaning, NLTK stopword removal), TF-IDF feature extraction, N-Gram analysis, model training (MultinomialNB), hyperparameter tuning via GridSearchCV, and interactive ipywidgets GUI testing.  
*spam_model_nb.pkl: Serialized, trained Naïve Bayes classifier ready for deployment and immediate inference.  
*tfidf_vectorizer.pkl: Serialized TF-IDF vectorizer used to transform new incoming raw text messages into the exact feature format expected by spam_model_nb.pkl.    
*spam_predictions_report.csv: Exported output report containing batch testing results, predicted labels, and model confidence percentages.   


Performance Summary

*Accuracy: 96.95%
*Mean 5-Fold Cross-Validation: 96.84% (± 0.78%)
*Best CV F1-Score (GridSearch): 94.15% (a = 0.1)
*ROC-AUC Score: 0.9789

Step-by-step process of the entire spam detection workflow:

1. Data Loading & Cleaning
Dataset Import (spam.csv): Loaded 5,572 labeled messages (~86.6% Ham, ~13.4% Spam).
Text Preprocessing: Converted text to lowercase, removed special characters/punctuation, and filtered out non-informative English stopwords using NLTK.
2. Feature Extraction & Engineering
TF-IDF Vectorization: Converted cleaned text into numerical term frequency-inverse document frequency matrices.
N-Gram Analysis: Incorporated bi-grams (ngram_range=(1, 2)) to capture word context and contiguous phrases, improving accuracy to 96.95%.
3. Model Training & Hyperparameter Tuning
Baseline Classifier: Trained a MultinomialNB (Naïve Bayes) model on the vectorized feature set.
Optimization (GridSearchCV): Fine-tuned the Laplace smoothing parameter (alpha = 0.1), achieving a peak Cross-Validation F1-Score of 94.15%.
4. Model Evaluation & Error Analysis
Cross-Validation: Ran 5-fold cross-validation (96.84% mean accuracy) to ensure model stability across unseen data.
ROC-AUC & Error Metrics: Plotted the ROC curve (AUC = 0.9789) and verified failure modes—achieving high precision with only 1 False Positive.
5. Artifact Export & Deployment Preparation
In-Notebook UI & Batch Inference: Built interactive testing widgets (ipywidgets) and exported batch prediction results with confidence scores to spam_predictions_report.csv.
Model Serialization: Saved spam_model_nb.pkl and tfidf_vectorizer.pkl to disk for integration into backend services (e.g., FastAPI in main.py).


Complete Notebook Code Execution & Output Breakdown

1. Model Evaluation & Performance Metrics
Code & Action: Loads spam.csv, renames columns to label and message, maps labels to numbers (ham: 0, spam: 1), cleans text using regex, performs an 80/20 train-test split, transforms text with TfidfVectorizer, and trains a MultinomialNB model.
Output: Achieves a baseline accuracy of 96.59% on the test set.
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/9ecfebe6-edc6-4baf-98da-d39fa4afe0d6" />

2. Model Comparison & Visualizations
Code & Action: Trains a LogisticRegression model for comparison, plots a Seaborn confusion matrix heatmap for Naïve Bayes, and defines a custom predict_message() function to test individual input sentences.
Output: Logistic Regression achieves 97.22% accuracy. The Naïve Bayes confusion matrix reveals 965 True Negatives, 1 False Positive, 37 False Negatives, and 112 True Positives. Custom test inputs correctly identify sample spam (84.45% confidence) and ham (99.63% confidence).
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/266ef497-43c0-4547-a8e7-c91e35b25f99" />

3. Model Serialization & Persistence
Code & Action: Uses joblib to serialize and save the trained Naïve Bayes classifier (spam_model.pkl) and TF-IDF vectorizer (tfidf_vectorizer.pkl) to disk.
Output: Exports pipeline .pkl binary files for deployment and future reuse without re-training.

4. Interactive User Interface
Code & Action: Initializes dynamic ipywidgets controls (Textarea, Button, Output) with a button click handler function (on_button_clicked) inside the notebook.
Output: Renders an interactive text box directly in the notebook allowing users to test custom messages and view real-time predictions with confidence percentages.
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/b1e12a7d-bbd9-401c-b443-a88a68f70646" />

5. Feature Importance & Top Predictive Words
Code & Action: Computes log likelihood ratios from model feature probabilities to isolate top predictive spam keywords (claim, prize, won, urgent, nokia, etc.) and displays a horizontal bar plot alongside overall precision, recall, and F1-score metrics.
Output: Highlights Precision: 99.12%, Recall: 75.17%, and F1-Score: 85.50% alongside a feature importance visualization.
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/95865633-e190-49af-a428-a59887ab9ab8" />

6. Batch Prediction & Report Export
Code & Action: Preprocesses a list of unseen sample messages, computes batch predictions and probability confidence scores, constructs a Pandas DataFrame, and saves it to disk via .to_csv().
Output: Generates a structured prediction output table and exports spam_predictions_report.csv.
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/1b4046a2-b89c-4f7e-b759-e8f7ccd581fe" />

7. Pipeline Artifacts & Project Summary
Code & Action: Uses joblib to serialize and save the model (spam_model_nb.pkl) and vectorizer (tfidf_vectorizer.pkl), prints the saved asset file sizes in KB, reloads them to run a sanity check prediction on a sample query ("URGENT! Your mobile number has won..."), and prints a project completion checklist.
Output: Saved spam_model_nb.pkl (94.52 KB) and tfidf_vectorizer.pkl (105.02 KB). Sanity check reload successfully predicted the sample test query as SPAM (99.09% confidence).
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/3bda0826-b503-47a4-8b80-28be82b5eca8" />

8. Advanced Feature Engineering: N-Gram Analysis
Code & Action: Initializes TfidfVectorizer with ngram_range=(1, 2) (capturing both unigrams and bigrams) limited to max_features=3000. Fits and transforms the dataset, then trains a new MultinomialNB model.
Output: Achieves an N-Gram Model Accuracy of 96.95%.
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/aae1ed4d-bda3-4ecb-8c5a-3c980cc38a37" />

9. Text Preprocessing with NLTK
Code & Action: Imports NLTK and downloads the standard English stopwords corpus to filter out low-value stop words.
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/5110c473-e913-4810-960c-ebb7d95d1c89" />

10. Dataset Class Distribution Analysis
Code & Action: Evaluates the class imbalance in the dataset using .value_counts(). Defines clean_text_with_stopwords() to apply lowercasing, regex stripping, and NLTK stop-word removal across df['message'].
Output: Reveals a dataset distribution of 4,825 Ham (86.59%) and 747 Spam (13.41%) messages.
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/7204ffcd-f673-45cb-9c07-cfd30626d0fa" />

11. Exploratory Data Analysis: Word Clouds
Code & Action: Generates comparative WordCloud visual representations for both classes. Joins cleaned spam messages into spam_text (black background) and ham messages into ham_text (white background), displaying them side-by-side using matplotlib.
Output: Renders word clouds illustrating high-frequency spam terms (free, text, txt, mobile, call, claim) versus ham terms (ok, come, dont, like, home).
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/47c0f1b8-aa41-4b4a-9b69-1a4751af3e40" />


Additional Implementation

12. Model Validation: K-Folds Cross Validation
Code & Action: Imports cross_val_score from sklearn.model_selection and runs a 5-fold cross-validation on the TF-IDF Naïve Bayes model across X_train_tfidf and y_train.
Output: Yields fold accuracies of [0.9742, 0.9619, 0.9787, 0.9574, 0.9697], resulting in a Mean CV Accuracy of 96.84% (± 0.78%).
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/d74fa68f-5cbb-4953-bdc6-b9619a5662db" />

13. Performance Evaluation: ROC Curve & AUC Score
Code & Action: Calculates predicted probability estimates (model.predict_proba) for the test set, computes roc_curve false-positive and true-positive rates, plots the ROC curve against a random-guess baseline using plt.plot(), and calculates the overall Area Under the Curve (roc_auc_score).
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/568f4678-29f7-4299-9296-acf95cc72d97" />

14. Hyperparameter Tuning: GridSearchCV
Code & Action: Imports GridSearchCV from sklearn.model_selection to optimize the Naïve Bayes smoothing parameter (alpha tested across [0.1, 0.5, 1.0, 2.0]) using 5-fold cross-validation.
Output Results:
Best Alpha Parameter: {'alpha': 0.1}
Best CV F1-Score: 94.15%
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/91db1653-7c8a-4ce8-9e74-84cbd22ab382" />

15. Error Analysis: False Positives & False Negatives
Code & Action: Constructs a Pandas DataFrame (test_analysis) pairing original test messages (X_test), actual labels (y_test), and predicted values (y_pred). Filters for misclassified instances to isolate False Positives (Actual: 0, Predicted: 1) and False Negatives (Actual: 1, Predicted: 0), displaying sample rows via .head(3).
Output Results:
Total False Positives (Ham flagged as Spam): 1
Sample: "nokia phone is lovly" (Triggered by the strong spam indicator keyword nokia).
Total False Negatives (Spam missed by filter): 37
Samples:
"you are now unsubscribed all services get tons..."
"freemsg hey there darling its been weeks now ..."
<img width="399" height="224" alt="image" src="https://github.com/user-attachments/assets/155ee534-8106-4dff-b590-a67e5af1653d" />


Conclusion

This project demonstrates an end-to-end Machine Learning pipeline for real-time text classification—from raw data preprocessing and N-Gram feature engineering to hyperparameter tuning and model serialization.
By leveraging a Naïve Bayes classifier (MultinomialNB), the system achieves 96.95% accuracy and a 0.9789 ROC-AUC score, maintaining high precision with minimal False Positives. With serialized pipeline artifacts (.pkl) and a lightweight FastAPI service (main.py), the model is fully equipped for real-world deployment, batch prediction reporting, and interactive inference.
