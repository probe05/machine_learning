import numpy as np
import matplotlib.pyplot as plt

def k_means(X,k=3, maxiter=100,tol=1e-4):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    labels = np.zeros(n_samples)
    id=np.random.choice(n_samples,k,replace=False)
    centroids = X[id].copy()
    x=X.reshape(n_samples, 1, n_features)
    for _ in range(maxiter):
        distances=np.sum((x-centroids)**2,axis=2)
        newLabels=np.argmin(distances, axis=1)
        if np.all(newLabels==labels):
            break
        labels=newLabels
        newCentroids=np.zeros((k,n_features))
        for i in range(k):
            a=X[newLabels==i]
            b=np.mean(a,axis=0)
            newCentroids[i]=b
        if np.linalg.norm(newCentroids-centroids)<tol:
            break
        centroids=newCentroids
    return labels,centroids

if __name__ == "__main__":
    np.random.seed(42)
    cluster1 = np.random.normal(loc=[2, 2], scale=0.5, size=(50, 2))
    cluster2 = np.random.normal(loc=[3, 3], scale=0.5, size=(50, 2))
    cluster3 = np.random.normal(loc=[4, 2], scale=0.5, size=(50, 2))
    X = np.vstack([cluster1, cluster2, cluster3])

    labels, centroids = k_means(X, k=3)

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=200, label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

    print("最终质心坐标：")
    print(centroids)