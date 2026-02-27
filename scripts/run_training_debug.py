import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.train_models import CyberbullyingModelTrainer

trainer = CyberbullyingModelTrainer()
print('Loaded trainer')
df = trainer.load_data()
print('Loaded data')
trainer.prepare_data(df)
print('Prepared split')

X_train_tfidf, X_test_tfidf = trainer.create_tfidf_features()
print('TF-IDF created')

m, p = trainer.train_naive_bayes(X_train_tfidf, X_test_tfidf)
print('Naive Bayes done')

m, p = trainer.train_svm(X_train_tfidf, X_test_tfidf)
print('SVM done')

m, p = trainer.train_logistic_regression(X_train_tfidf, X_test_tfidf)
print('Logistic Regression done')

m, p = trainer.train_random_forest(X_train_tfidf, X_test_tfidf)
print('Random Forest done')

try:
print('XGBoost skipped by configuration or not available')

m, p = trainer.train_ensemble(X_train_tfidf, X_test_tfidf)
print('Ensemble done')

print('All training steps completed')
