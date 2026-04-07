#=============================Import============================#
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
#===============================================================#

#-------------------Loading-Data-Set----------------------------#
diabetes=load_diabetes()
X,y=diabetes.data,diabetes.target
#---------------------------------------------------------------#

#-------------------Split-Train-Test----------------------------#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=999)
#---------------------------------------------------------------#

def main():
    #----------------Model-Select-Random-Forest------------------------------------------#
    clf = RandomForestRegressor(n_estimators=100,max_depth=None,random_state=999)

    #-------------------------Training-Model---------------------------------------------#
    clf.fit(X_train,y_train)

    #--------------------------Predicting------------------------------------------------#
    y_pred = clf.predict(X_test)
    #---------------------------printing-output-score------------------------------------#
    print('r2_score')
    print(r2_score(y_test,y_pred))
    print('mean_squared_error')
    print(mean_squared_error(y_test,y_pred))

if __name__ == '__main__':
    main()