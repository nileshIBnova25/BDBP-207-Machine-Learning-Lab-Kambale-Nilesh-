# 3.Try classifying classes 1 and 2 from the iris dataset with SVMs, with the 2 first features.
# Leave out 10% of each class and test prediction performance on these observations.
# https://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html#super
# vised-learning-tut - Check the solution code to learn about various plots

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

#---------------Loading-Data------------------------#
iris = datasets.load_iris()
X_df = iris.data
y_df = iris.target
#----------------------------------------------------#


def main():

    #---------------Filter for Class 0 (Setosa) and Class 1 (Versicolor)-------
    mask = y_df < 2
    X = X_df[mask, :2]
    y = y_df[mask]

    #----- 2. Manual 10% hold-out (5 samples per class of 50)------------
    np.random.seed(42)    # We use a fixed seed for reproducibility)
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    #-------------Split: 90 samples for training, 10 for testing--------------
    X_train, X_test = X[:90], X[90:]
    y_train, y_test = y[:90], y[90:]

    #-------------------- 3. Fit SVM model-----------------------------------
    # C is the regularization parameter; a high C means a hard margin
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X_train, y_train)

    #-----------------4. Predict and Evaluate--------------------------------------
    prediction = clf.predict(X_test)
    accuracy = np.mean(prediction == y_test) * 100

    print(f"Test Accuracy: {accuracy}%")
    print(f"Predictions: {prediction}")
    print(f"Actual:      {y_test}")


if __name__ == "__main__":
    main()