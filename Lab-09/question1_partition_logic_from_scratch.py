#--------------------------------------------------------------------#
#------------------------IMPORT--------------------------------------#
import pandas as pd
#--------------------------------------------------------------------#


#---------------------------Loading-Data-----------------------------#
df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
print(df.column)
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#

def partitioning_data(df,column,t):
    df_left=df[df[column] <= t ].reset_index(drop=True)
    df_right=df[df[column] > t ].reset_index(drop=True)
    return df_left,df_right



def main():
    thr = [78,80,82]
    for t in thr:
        df_left,df_right = partitioning_data(df,column='BP',t=t)
        print(f'Partitioning with split value:  {t} ')
        print("left child data;\n",df_left)
        print("right child data\n",df_right)
        print('=' * 80)

if __name__== "__main__":
    main()



# def partitioning_logi(df,col_name,split_val):
#     idx=df['BP'].index()
#     feature_val=df['BP'].value.tolist()
#     df_left=[]
#     df_right=[]
#     for i in range(len(feature_val)):
#         if  feature_val[i] > 80 :
#             df_right.append(feature_val[i])
#         else:
#             df_left.append(feature_val[i])










