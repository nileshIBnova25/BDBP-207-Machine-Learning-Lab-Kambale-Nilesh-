import pandas as pd
import math
from sklearn.datasets import load_iris
from collections import Counter

#----------------------Loading Data--------------------------------#
iris=load_iris(as_frame=True)
df=iris.frame
print(df.columns)
# import seaborn as sns
# df = sns.load_dataset('iris')  #alternate way of Loading
# print(df.head(150))
#-------------------------------------------------------------------#

cols = df.columns.tolist()
def entropy(label):
    total = len(label)
    label_counts = Counter(label)
    e=0
    for count in label_counts.values():
        prob = count/total
        e -= prob * math.log2(prob)
    return e

def sub_set(df,k):
    child=[]
    column = df.iloc[:,k]
    for value in column.unique():
        subset = df[column == value].iloc[:,-1].tolist()
        child.append(subset)
    return child



def weighted_entropy(df,k):
    len_parent = len(df)
    child = sub_set(df,k)
    e_weighted = 0
    for sub in child:
        e_weighted += (len(sub)/len_parent) * entropy(sub)
    return e_weighted


def main():
    col= df.iloc[:,-1]
    #print(col)
    parent_e=entropy(col)
    print(parent_e)
    cols_no=df.shape[1]
    for i in range(cols_no - 1):
        w_entropy_of_child = weighted_entropy(df,i)
        print("---" * 20)
        print( f"\nweighted entropy for feature/column no {i+1}",w_entropy_of_child)
        print(f"\ninformation gain for feature/column No {i+1} ", parent_e - w_entropy_of_child )

if __name__ == "__main__":
    main()



