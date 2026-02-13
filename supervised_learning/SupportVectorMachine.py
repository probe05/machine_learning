import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
class SVM_SMO:
    def __init__(self,C,sigma,maxIter=100,tol=1e-3):
        self.C=C
        self.sigma=sigma
        self.maxIter = maxIter
        self.tol = tol

        self.X=None
        self.y=None
        self.alpha=None
        self.K=None
        self.b=0.0
        self.ECache=None

    def train(self,X,y):
        n=len(X)
        self.X=X
        self.y=y
        self.K = self.calcKMat()
        self.alpha=np.zeros(n)
        self.ECache=np.zeros((n,2))

        iterNum=0
        changeNum=0
        while iterNum<self.maxIter:
            bestI=self.selectI()
            if bestI ==-1:
                print("无违反KKT点")
                break
            changed = self.smo(bestI)
            if changed:
                changeNum+=1
            iterNum+=1
            if iterNum%10==0:
                print(f'已迭代{iterNum}次,优化参数对{changeNum}对')
        print(f'最终已完成优化{iterNum}次,优化参数对{changeNum}对')
        return self
    def smo(self,i):
        Ei=self.ECache[i][1]
        yi = self.y[i]
        Eyi = Ei * yi
        if (Eyi < -self.tol and self.alpha[i] < self.C) or (Eyi > self.tol and self.alpha[i] > 0):
            j,Ej=self.selectJ(i)
            alpha_i_old = self.alpha[i].copy()
            alpha_j_old = self.alpha[j].copy()

            if self.y[i] != self.y[j]:
                L = max(0, alpha_j_old - alpha_i_old)
                H = min(self.C, self.C + alpha_j_old - alpha_i_old)
            else:
                L = max(0, alpha_j_old + alpha_i_old - self.C)
                H = min(self.C, alpha_j_old + alpha_i_old)
            if L == H:
                return False

            eta = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]
            if eta <= 0:
                return False

            self.alpha[j] += self.y[j] * (Ei - Ej) / eta
            self.alpha[j] = self.clip_alpha(self.alpha[j], H, L)
            self.update_Ei(j)

            if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                return False

            self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])
            self.update_Ei(i)

            b1 = self.b - Ei - self.y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i] - self.y[j] * (
                        self.alpha[j] - alpha_j_old) * self.K[i, j]
            b2 = self.b - Ej - self.y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j] - self.y[j] * (
                        self.alpha[j] - alpha_j_old) * self.K[j, j]
            if 0 < self.alpha[i] < self.C:
                self.b = b1
            elif 0 < self.alpha[j] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2

            return True
        return False
    def clip_alpha(self, aj, H, L):
        if aj > H:
            aj = H
        if aj < L:
            aj = L
        return aj

    def update_Ei(self, i):
        Ei = self.calcEi(i)
        self.ECache[i] = [1, Ei]
    def selectJ(self, i):
        maxDeltaE = -np.inf
        bestJ = -1
        bestEj = 0
        for j in range(len(self.alpha)):
            if j == i:
                continue
            Ei = self.ECache[i][1]
            Ej = self.ECache[j][1]
            delta_E = abs(Ei - Ej)
            if delta_E > maxDeltaE:
                maxDeltaE = delta_E
                bestJ = j
                bestEj = Ej
        return bestJ,bestEj
    def selectI(self):
        n=len(self.X)
        violateKKTIndex=[]
        violateKKTvalue=[]
        for i in range(n):
            Ei = self.calcEi(i)
            yi=self.y[i]
            self.ECache[i]=[1,Ei]
            Eyi=Ei*yi
            if (Eyi<-self.tol and self.alpha[i]<self.C) or (Eyi>self.tol and self.alpha[i]>0):
                violateKKTIndex.append(i)
                violateKKTvalue.append(abs(Eyi))
        if len(violateKKTIndex) == 0:
            return -1
        i = np.argmax(violateKKTvalue)
        bestI = violateKKTIndex[i]
        return bestI
    def calcEi(self, i):
        fxi = np.sum(self.alpha * self.y * self.K[i]) + self.b
        Ei = fxi - self.y[i]
        return Ei
    def calcKMat(self):
        n=len(self.X)
        K=np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                K[i][j]=self.calcK(self.X[i],self.X[j])
        return K
    def calcK(self,xi,xj):
        return np.exp(-np.linalg.norm(xi-xj)**2/(2*self.sigma**2))

    def predict_single(self, x):
        fx = 0.0
        support_vec_idx = np.where(self.alpha > 1e-5)[0]
        for j in support_vec_idx:
            fx += self.alpha[j] * self.y[j] * self.calcK(x, self.X[j])
        fx += self.b
        return np.sign(fx)

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])
if __name__ == "__main__":
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=10, noise=0.15, random_state=42)
    y = np.where(y == 0, -1, 1)
    svm=SVM_SMO(C=10,sigma=0.5)
    svm.train(X,y)

    # 预测并计算准确率
    y_pred = svm.predict(X)
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"训练集准确率：{accuracy:.4f}")
    print(svm.alpha)
    # 可视化
    def plot_decision_boundary(model, X, y):
        h = 0.02
        x_min, x_max = X[:, 0].min()-0.5, X[:, 0].max()+0.5
        y_min, y_max = X[:, 1].min()-0.5, X[:, 1].max()+0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
        plt.figure(figsize=(8,6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
        plt.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.8)
        plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap_bold, edgecolors='k', s=50)
        support_vec_idx = np.where(model.alpha > 1e-5)[0]
        plt.scatter(X[support_vec_idx,0], X[support_vec_idx,1], s=150, facecolors='none', edgecolors='k', label='Support Vectors')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('SVM SMODecision Boundary')
        plt.legend()
        plt.show()

    plot_decision_boundary(svm, X, y)