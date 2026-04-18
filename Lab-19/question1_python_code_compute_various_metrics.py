#---------------------Import-----------------------------------#
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
#---------------------------------------------------------------#

#-------------Loding-Data---------------------------------------#
df = pd.read_csv('Heart.csv').drop(columns=['Unnamed: 0'], errors='ignore').dropna()
df['AHD'] = df['AHD'].map({'Yes': 1, 'No': 0})

X = df.drop(columns=['AHD'])
y = df['AHD']

categorical_features = ['ChestPain', 'Thal']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough' # Leave numeric columns alone
)

X_encoded = preprocessor.fit_transform(X)
#---------------------------------------------------------------#

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

#-----------------model-for-evaluating---------------------------#
def model_eval(X_train, X_test, y_train, y_test):

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scl = scaler.transform(X_train)
    X_test_scl = scaler.transform(X_test)
    clf = LogisticRegression()
    clf.fit(X_train_scl, y_train)
    y_pred = clf.predict(X_test_scl)

    return y_pred.tolist() , clf , X_test_scl
#-----------------------------------------------------------------#
#---------------------Counter-------------------------------------#
def counter_(y_test, y_pred):
    tp,tn,fp,fn = 0,0,0,0
    for i in range(len(y_test)):

        if y_test[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_test[i] == 0 and y_pred[i] == 0:
            tn += 1
        elif y_test[i] == 1 and y_pred[i] == 0:
            fn += 1
        else:
            fp +=1

    return tp,tn,fp,fn
#-----------------------------------------------------------------#


#----------------------Accuracy-----------------------------------#
def accuracy_(y_t, y_p):
    tp,tn,fp,fn = counter_(y_t, y_p)
    return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0

#-----------------------------------------------------------------#
#---------------------Precision---------------------------------#
def precision_(yt, yp):
    tp,tn,fp,fn = counter_(yt,yp)
    return tp / (tp + fp) if (tp + fp) != 0 else 0
#-----------------------------------------------------------------#


#---------------------Sensitivity---------------------------------#
def sensitivity_(yt, yp):
    tp,tn,fp,fn = counter_(yt,yp)
    return tp / (tp + fn) if (tp + fn) != 0 else 0
#-----------------------------------------------------------------#

#---------------------Specificity---------------------------------#
def specificity_(yt, yp):
    tp,tn,fp,fn = counter_(yt,yp)
    return tn / (tn + fp) if (tn + fp) != 0 else 0
#-----------------------------------------------------------------#

#---------------------F1-Score------------------------------------#
def f1_score_(yt, yp):
    tp,tn,fp,fn = counter_(yt,yp)
    return (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
#-----------------------------------------------------------------#


#--------------Compute_ROC-----------------------------------------#
import numpy as np

def compute_roc(y_test, X_test_scl, model, step):

    # probabilities for class 1
    y_prob = model.predict_proba(X_test_scl)[:, 1]

    # convert once
    y_test = y_test.values.tolist()

    tpr_list = []
    spc_list = []
    fpr_list = []

    # threshold sweep
    for thr in np.linspace(0, 1, step):

        y_pred = (y_prob >= thr).astype(int).tolist()

        tpr_list.append(sensitivity_(y_test, y_pred))
        spc_list.append(specificity_(y_test, y_pred))
        fpr_list.append(1 - specificity_(y_test, y_pred))

    return tpr_list, spc_list, fpr_list
#------------------------------------------------------------------#

#------------------------------------Roc_auc-----------------------#
def compute_auc(fpr, tpr):

    fpr = np.array(fpr)
    tpr = np.array(tpr)

    # sort by FPR (VERY IMPORTANT)
    idx = np.argsort(fpr)
    fpr = fpr[idx]
    tpr = tpr[idx]

    auc_value = 0

    for i in range(1, len(fpr)):
        auc_value += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2

    return auc_value
#------------------------------------------------------------------#

#----------------------plot-function-------------------------------#
def plot_roc_curve(fpr, tpr, auc_val):

    plt.figure(figsize=(8, 6))

    # ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {auc_val:.2f})')

    # diagonal (random classifier)
    plt.plot([0, 1], [0, 1], linestyle='--', color='blue')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.show()

#------------------------------------------------------------------------------#


def main():
    y_test_ = y_test.tolist()
    y_pred ,model ,X_test_scl = model_eval(X_train, X_test, y_train, y_test)

    #print(accurac(y_test_, y_pred))
    print(f'Accuracy:{accuracy_(y_test_, y_pred)}')
    print(f'Precision:{precision_(y_test_, y_pred)}')
    print(f"Sensitivity :{sensitivity_(y_test_, y_pred)}")
    print(f"Specificity :{specificity_(y_test_, y_pred)}")
    print(f"F1 score :{f1_score_(y_test_, y_pred)}")
    print(classification_report(y_test_, y_pred))

    tpr,spc,fpr = compute_roc(y_test,X_test_scl,model,1000)
    auc=compute_auc(fpr,tpr)

    plot_roc_curve(fpr,tpr,auc)

    print(f"AUC:{auc}")

    specificity = specificity_(y_test_, y_pred)

if __name__ == '__main__':
    main()

