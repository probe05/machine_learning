import numpy as np
import matplotlib.pyplot as plt
def standardize(X):
    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    x_std[x_std == 0] = 1e-8
    return (X-x_mean)/x_std
def PCA(X,k):
    X=standardize(X)
    covMat = np.dot(X.T, X)/(X.shape[0]-1)
    eigvalues,eigvectors = np.linalg.eig(covMat)
    sorted_idx = np.argsort(eigvalues)[::-1]
    eigvectors = eigvectors[:,sorted_idx]
    eigvalues = eigvalues[sorted_idx]
    explainedVarRatio=[]
    tmp=0
    for i,v in enumerate(eigvalues):
        tmp=tmp+v
        explainedVarRatio.append(tmp/(eigvalues.sum()))
    print(explainedVarRatio)
    print(eigvectors)
    projectedMat = eigvectors[:,:k]
    print(projectedMat)
    X_pca = np.dot(X, projectedMat)
    return X_pca
if __name__ == '__main__':
    n_samples=100
    feat1 = np.random.normal(loc=0, scale=1, size=n_samples)
    feat2 = 0.8 * feat1 + np.random.normal(loc=0, scale=0.5, size=n_samples)
    feat3 = 0.5 * feat1 + 0.3 * feat2 + np.random.normal(loc=0, scale=0.4, size=n_samples)
    x_3d = np.column_stack([feat1, feat2, feat3])
    x_pca = PCA(x_3d,2)
    fig = plt.figure(figsize=(10, 8))
    ax1=fig.add_subplot(121,projection='3d')
    ax1.scatter(x_3d[:,0],x_3d[:,1],x_3d[:,2], alpha=0.6,c='b')
    ax2=fig.add_subplot(122)
    ax2.scatter(x_pca[:,0],x_pca[:,1], alpha=0.6,c='r')
    plt.show()