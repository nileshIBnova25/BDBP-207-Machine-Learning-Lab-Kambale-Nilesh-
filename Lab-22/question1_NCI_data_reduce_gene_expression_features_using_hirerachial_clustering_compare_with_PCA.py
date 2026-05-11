#------------------------------Import-----------------------------------------------#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ISLP import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.cluster.hierarchy import linkage , dendrogram
#-----------------------------------------------------------------------------------#

#----------------------------------------Load-Data----------------------------------#
data = load_data("NCI60")
X = data["data"]
y = data["labels"]

print("Dataset Shape: ", X.shape)
print("Number of Classes: ", len(np.unique(y)))
#-----------------------------------------------------------------------------------#

#-------------------------------------Data-Preprocessing----------------------------#
X = np.nan_to_num(X,nan=np.nanmean(X))  # filling missing values
y=y.values.ravel()
scaler = StandardScaler() # Standardization
X_scaled = scaler.fit_transform(X)
#-----------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------#

def feature_hierarchical_clustering_plot(X_scaled,n_clusters=50):
    print("\nPerforming Hierarchical Clustering")
    X_features = X_scaled.T       # Because we want to cluster by features/gene Transpose Matrix
    subset = X_features[:200]     # Large sample difficult to visualize
    z = linkage(subset, method='ward')
    plt.figure(figsize=(14,6))
    dendrogram(z)

    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Features/Genes")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

    # feature clustering
    hc = AgglomerativeClustering(n_clusters=n_clusters,
                                 metric='euclidean',
                                 linkage='ward')
    feature_clusters= hc.fit_predict(X_features)

    selected_features = []

    for cluster_id in np.unique(feature_clusters):  # select one representative from each cluster
        feature_index = np.where(feature_clusters == cluster_id)[0][0]

        selected_features.append(feature_index)

    # Reduced dataset
    X_hc = X_scaled[:, selected_features]
    print("Original Shape : ", X_scaled.shape)
    print("HC Reduced Shape :", X_hc.shape)

    return X_hc


def apply_pca(X_scaled,n_components):

    print("\nApplying PCA ...\n")

    pca = PCA(n_components=n_components)

    X_pca = pca.fit_transform(X_scaled)

    print("PCA Reduced Shape :", X_pca.shape)

    return X_pca

def knn_classification(X_train_pca,X_test_pca,y_train,y_test,method_name):

    knn=KNeighborsClassifier(n_neighbors=3)

    knn.fit(X_train_pca,y_train)

    y_pred = knn.predict(X_test_pca)

    accuracy = accuracy_score(y_test,y_pred)

    print(f"{method_name} Accuracy : {accuracy:.4f}")

    return accuracy

#
# def main():
#     X_hc=feature_hierarchical_clustering_plot(X_scaled)
#
#     X_train_hc,X_test_hc,y_train_hc,y_test_hc = train_test_split(X_hc,y,test_size=0.2,random_state=42,stratify=y)
#     hc_accuracy=knn_classification(X_train_hc,X_test_hc,y_train_hc,y_test_hc,"hierarchical_clustering")
#
#     X_pca=apply_pca(X_scaled,20)
#     X_train_pca,X_test_pca,y_train_pca,y_test_pca=train_test_split(X_pca,y,test_size=0.2,random_state=42,stratify=y)
#
#     pca_accuracy=knn_classification(X_train_pca,X_test_pca,y_train_pca,y_test_pca,"PCA_hierarchical_clustering")





def main():

    # Hierarchical Clustering
    X_hc = feature_hierarchical_clustering_plot(
        X_scaled,
        n_clusters=50
    )

    X_train_hc, X_test_hc, y_train_hc, y_test_hc = train_test_split(
        X_hc,
        y,
        test_size=0.2,
        random_state=42,
    )

    hc_accuracy = knn_classification(
        X_train_hc,
        X_test_hc,
        y_train_hc,
        y_test_hc,
        "Hierarchical Clustering"
    )

    # PCA
    X_pca = apply_pca(X_scaled,20)

    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_pca,
        y,
        test_size=0.2,
        random_state=42,
    )

    pca_accuracy = knn_classification(
        X_train_pca,
        X_test_pca,
        y_train_pca,
        y_test_pca,
        "PCA"
    )

    # Final Comparison
    print("\n==============================")
    print("FINAL COMPARISON")
    print("==============================")

    print(f"Hierarchical Clustering Accuracy : {hc_accuracy:.4f}")
    print(f"PCA Accuracy                     : {pca_accuracy:.4f}")




if __name__ == '__main__':
    main()














