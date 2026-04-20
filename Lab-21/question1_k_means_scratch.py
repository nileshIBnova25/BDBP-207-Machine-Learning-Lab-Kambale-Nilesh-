import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def initialize_centroids(X, K):
    indices = np.random.choice(X.shape[0], K, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    distances = []
    for c in centroids:
        dist = np.linalg.norm(X - c, axis=1)
        distances.append(dist)
    distances = np.array(distances)
    return np.argmin(distances, axis=0)

def update_centroids(X, labels, K):
    new_centroids = []
    for k in range(K):
        points = X[labels == k]
        if len(points) == 0:
            new_centroids.append(X[np.random.randint(0, X.shape[0])])
        else:
            new_centroids.append(points.mean(axis=0))
    return np.array(new_centroids)

def kmeans(X, K=3, max_iters=100):
    centroids = initialize_centroids(X, K)
    for i in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, K)

        if np.allclose(centroids, new_centroids):
            print(f"Converged at iteration {i+1}")
            break

        centroids = new_centroids

    return centroids, labels

X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

centroids, labels = kmeans(X, K=3)

plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.scatter(centroids[:,0], centroids[:,1], color='red', s=200)
plt.show()