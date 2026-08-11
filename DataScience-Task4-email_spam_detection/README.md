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

1. Library Imports & Setup
Code: Imports data manipulation (pandas, numpy), visualization (matplotlib, seaborn), text processing (nltk), and modeling modules (sklearn).
Outputs: Sets up the execution environment and downloads required NLTK tokenizers and stopword corpora.

2. Dataset Ingestion
Code: Reads spam.csv into a Pandas DataFrame using latin-1 encoding and inspects initial dataset dimensions.
Outputs: Confirms raw dataset size of 5,572 rows and 5 initial columns (v1 through v5).

3. Data Cleaning & Renaming
Code: Drops unused metadata columns (Unnamed: 2, Unnamed: 3, Unnamed: 4) and renames v1 \rightarrow label and v2 \rightarrow message.
Outputs: Clean DataFrame containing only the core target and text message columns.

4. Label Encoding
Code: Maps string class labels into binary numerical values (0 for ham and 1 for spam).
Outputs: Populates a structured numerical target column required by Scikit-Learn estimators.

<img width="1600" height="420" alt="image" src="https://github.com/user-attachments/assets/b1e12a7d-bbd9-401c-b443-a88a68f70646" />


6. Class Distribution Analysis
Code: Computes absolute counts and percentage distributions for both classes using .value_counts().
Outputs: Identifies class imbalance: 4,825 Ham (86.59%) vs. 747 Spam (13.41%) messages.

7. Text Length Profiling
Code: Calculates message character lengths, word counts, and sentence counts for ham and spam categories.
Outputs: Statistical tables demonstrating that spam messages carry higher average word counts and character lengths.

8. Visual Exploratory Analysis
Code: Plots class distribution bar charts, message length histograms, and generates WordClouds for each class.
Outputs: Visual confirmation of prominent spam terms (e.g., "call", "free", "claim", "txt") versus standard conversational ham vocabulary.

9. Custom Text Preprocessing
Code: Defines a cleaning pipeline to lowercase text, strip punctuation and special characters, tokenize words, and eliminate NLTK English stopwords.
Outputs: Generates a sanitized clean_message column free of noise and non-informative words.

10. Stratified Train-Test Split
Code: Executes train_test_split with stratify=y and random_state=42 to create an 80/20 train-test partition.
Outputs: Splits dataset into 4,457 training samples and 1,115 test samples while preserving target class proportions.

11. TF-IDF & N-Gram Feature Extraction
Code: Fits TfidfVectorizer with ngram_range=(1, 2) on training data and transforms test data.
Outputs: High-dimensional sparse feature matrix capturing both individual terms and key contiguous word pairs (e.g., "claim now", "free msg").

12. Baseline Naïve Bayes Fitting
Code: Trains a default MultinomialNB classifier on the TF-IDF feature matrix and evaluates test set predictions.
Outputs: Initial baseline model accuracy of 96.95%.

13. Hyperparameter Tuning (GridSearchCV)
Code: Runs GridSearchCV across Laplace smoothing parameters (alpha: [0.1, 0.5, 1.0, 2.0]) using 5-fold cross-validation targeting F1-Score.
Outputs: Selects optimal parameter {'alpha': 0.1} yielding a peak Cross-Validation F1-Score of 94.15%.

14. Cross-Validation & Stability Evaluation
Code: Evaluates 5-fold cross_val_score across the dataset using the tuned Naïve Bayes estimator.
Outputs: Mean 5-Fold Cross-Validation Accuracy of 96.84% (\pm 0.78\%), confirming stability across different data folds.

15. ROC Curve & AUC Score Calculation
Code: Computes prediction probabilities, generates the Receiver Operating Characteristic (ROC) curve using roc_curve, and computes roc_auc_score.
Outputs: ROC-AUC Score of 0.9789, showing class separation capability across probability thresholds.

16. Misclassification Inspection & Interactive UI
Code: Isolates False Positives and False Negatives, exports prediction outputs to spam_predictions_report.csv, serializes model files (.pkl), and sets up dynamic ipywidgets UI controls.
Outputs:
Error Breakdown: 1 False Positive and 37 False Negatives.
Interactive text input widget providing real-time spam predictions and confidence percentages.


Conclusion

This project demonstrates an end-to-end Machine Learning pipeline for real-time text classification—from raw data preprocessing and N-Gram feature engineering to hyperparameter tuning and model serialization.
By leveraging a Naïve Bayes classifier (MultinomialNB), the system achieves 96.95% accuracy and a 0.9789 ROC-AUC score, maintaining high precision with minimal False Positives. With serialized pipeline artifacts (.pkl) and a lightweight FastAPI service (main.py), the model is fully equipped for real-world deployment, batch prediction reporting, and interactive inference.
