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

Conclusion

This project demonstrates an end-to-end Machine Learning pipeline for real-time text classification—from raw data preprocessing and N-Gram feature engineering to hyperparameter tuning and model serialization.
By leveraging a Naïve Bayes classifier (MultinomialNB), the system achieves 96.95% accuracy and a 0.9789 ROC-AUC score, maintaining high precision with minimal False Positives. With serialized pipeline artifacts (.pkl) and a lightweight FastAPI service (main.py), the model is fully equipped for real-world deployment, batch prediction reporting, and interactive inference.
