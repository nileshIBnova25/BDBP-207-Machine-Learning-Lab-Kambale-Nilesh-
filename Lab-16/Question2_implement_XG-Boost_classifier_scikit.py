#-------------------Imports----------------#
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report

#---------------------------Load-data-----------------------#
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

#------------Train-Test-Split-------------------------------------#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    # Initialize Classifier
    clf_model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.2,
        eval_metric='logloss'
    )

    # Fit and Predict
    clf_model.fit(X_train, y_train)
    y_pred = clf_model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()