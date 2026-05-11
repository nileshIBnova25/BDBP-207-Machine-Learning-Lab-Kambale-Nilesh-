import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

def intialize_centroids(X,K):
    indices = np.random.choice(X.shape[0], K,replace = False)
    return X[indices]

def assign_clusters(X,centroids):
    distances = []
    for i in centroids:
        distances.append(np.linalg.norm(X - i ,axis=1))
    return np.argmin(distances, axis=0)

def update_centroids(X,labels ,K):
    n_centroids = []
    for i in range(K):
        clust = X[labels == i]
        if len(clust) == 0:
            n_centroids.append(np.random.randint(0,X.shape[0]))
        else:
            n_centroids.append(np.mean(clust,axis=0))
    return np.array(n_centroids)

def kmeans(X,K,max_iter):
    centroids = intialize_centroids(X,K)
    for i in range(max_iter):
        clust_label = assign_clusters(X,centroids)
        new_centroids=update_centroids(X,clust_label,K)

        if np.allclose(centroids,new_centroids):
            print(f"Converged at iteration {i+1}")
            break

        centroids = new_centroids
    return centroids,clust_label

def main():
    X, y = make_blobs(n_samples=1000,centers=10,random_state=42)
    centroids,clust_label = kmeans(X,8,100)

    plt.scatter(X[:, 0],X[:, 1],c=clust_label,cmap='viridis',s=8)
    plt.scatter(centroids[:, 0],centroids[:, 1],c='red',s=50)
    plt.show()

if __name__ == '__main__':
    main()




