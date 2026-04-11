import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score , classification_report

#-----------------------IMPORT---------------------------#
df = pd.read_csv('Weekly.csv')

#------------------------Split---------------------------#
X = df.drop(['Direction', 'Today'], axis=1)
y = np.where(df['Direction'] == 'Up', 1, 0)

#------------------------Train-test-split----------------#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    #---------------Model-------------------------#
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=5,
        random_state=42

    )
    #-------------------Training------------------#
    model.fit(X_train, y_train)
    #--------------------Predicting---------------#
    y_pred = model.predict(X_test)
    #---------------------Printing----------------#
    print(f"Classification Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))
if __name__ == '__main__':
    main()