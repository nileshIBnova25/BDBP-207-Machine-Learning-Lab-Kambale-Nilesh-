import pandas as pd
import numpy as np

df = pd.read_csv("breast-cancer_2.csv")
col = ['age','menopause','tumor_size','inv_nodes','node_caps','deg_malig',
       'breast_r_l','quadrant_of_breast','irradiation','class']
df.columns = col

def one_hot_enc_col(column):
    unique_vals = sorted(column.dropna().unique())
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    one_hot_matrix = np.zeros((len(column), len(unique_vals)), dtype=int)
    #print(one_hot_matrix)
    for i, val in enumerate(column):
        if pd.notna(val):
            idx = mapping[val]
            one_hot_matrix[i, idx] = 1
    col_names = [f"{column.name}_{val}" for val in unique_vals]
    return pd.DataFrame(one_hot_matrix, columns=col_names), mapping

def encoded_data(df, columns):
    mappings = {}
    df_new = df.copy()
    for column in columns:
        one_hot_df, mapping = one_hot_enc_col(df[column])
        df_new = pd.concat([df_new, one_hot_df], axis=1)
        mappings[column] = mapping
    return df_new, mappings

def main():
    categorical_cols = ['age','menopause','tumor_size','inv_nodes','node_caps',
                        'deg_malig','breast_r_l','quadrant_of_breast','irradiation','class']
    df_encoded, mappings = encoded_data(df, categorical_cols)
    print(df_encoded.head())
    for col, map_val in mappings.items():
        print(f"{col}: {map_val}")

if __name__ == "__main__":
    main()