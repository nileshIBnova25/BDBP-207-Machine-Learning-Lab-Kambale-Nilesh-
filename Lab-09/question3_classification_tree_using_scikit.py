import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

#------------------------Loading Data ------------------------------#
# Make sure you have sonar dataset CSV, e.g., 'sonar.csv'
df = pd.read_csv('sonar.csv')
print("Columns in data:", df.columns.tolist())
print(df.head())

#-----------------------------Separating Features & target--------------------------------------#
# Assuming the last column is the target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]   # 'R' or 'M'

#-----------------------------Main Function--------------------------------------#
def main():
    #----------------------Splitting Into Train-Test----------------------------------------------#
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    #----------------------Training Classification Model----------------------------------------------#
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    #-------------------------Making Prediction-------------------------------------------------------#
    y_pred = model.predict(X_test)

    #---------------------------Evaluate Model-------------------------#
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    #---------------------------Visualize Decision Tree----------------#
    plt.figure(figsize=(20,10))
    plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True, rounded=True)
    plt.show()

#-----------------------------Run--------------------------------------#
if __name__ == "__main__":
    main()