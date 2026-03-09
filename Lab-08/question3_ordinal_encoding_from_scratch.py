#-------------Ordinal Encoding Using Scratch-------------------#
import pandas as pd
#--------------------------------------------------------------#

#-----------------------Loading Data-------------------------#
df = pd.read_csv("breast-cancer_2.csv")
print(df)
print(df.info())

# Correct column names
col = ['age','menopause','tumor_size','inv_nodes','node_caps','deg_malig',
       'breast_r_l','quadrant_of_breast','irradiation','class']
df.columns = col
print(df)
print(df.info())
#------------------------------------------------------------#

def ordinal_enc_col(column):
    unique_cate = sorted(column.dropna().unique())
    mapp = {category: i for i, category in enumerate(unique_cate)}
    enc_val = [mapp[k] if pd.notna(k) else -1 for k in column]
    return enc_val, mapp

def encoded_data(df, columns):
    enc_df = {}
    for column in columns:
        enc_col, mapp = ordinal_enc_col(df[column])
        df[column] = enc_col       # fix: use 'column' instead of 'col'
        enc_df[column] = mapp
    return df, enc_df

def main():
                                                    # Encode a single column for demo
    enc_val, mapp = ordinal_enc_col(df['age'])
    print(enc_val[:10])
    print(mapp)

                                                    # Encode all columns
    categorical_cols = ['age','menopause','tumor_size','inv_nodes','node_caps',
                        'deg_malig','breast_r_l','quadrant_of_breast','irradiation','class']
    df_encoded, mappings = encoded_data(df, categorical_cols)
    print(df_encoded.head(20))
    print(mappings)

if __name__ == "__main__":
    main()




