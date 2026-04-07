#===================Import=========================================#
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , accuracy_score
#==================================================================#

#----------Loading-data---------------#
iris = load_iris()
X = iris.data
y = iris.target

#----------------Spliting-Data-Train-Test----------------#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    #------------------creating object for model-----------------#
    clf = AdaBoostClassifier()

    #-----------------------Training model-----------------------#
    clf.fit(X_train, y_train)

    #-----------------------Predicting---------------------------#
    y_pred = clf.predict(X_test)

    #-------------------------Print------------------------------#
    print(f'accuracy')
    print({accuracy_score(y_test, y_pred)})
    print('classification report')
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
