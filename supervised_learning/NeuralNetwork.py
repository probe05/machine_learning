import numpy as np
import matplotlib.pyplot as plt
class NeuralNetwork:
    def __init__(self,layers,maxiter=500,eta=0.1):
        self.num_layers=len(layers)
        self.layers = layers
        self.maxiter=maxiter
        self.eta=eta

    def train(self,X,y):
        self.weights,self.biases = self.init_weights(X)
        self.loss_history=[]
        for epoch in range(self.maxiter):
            loss=self.loss_function(X,y)
            self.loss_history.append(loss)
            self.backward(X,y)
            self.update_params()
            if epoch % 1 == 0:
                print(f'Iteration {epoch+1}/{self.maxiter} Loss: {loss:.4f}')
    def predict(self,X):
        out=self.forward(X)
        ans=np.argmax(out,axis=0)
        return ans
    def backward(self,X,y):
        self.deltas={}
        delta_latter = self.forward(X)-y
        self.deltas[self.num_layers-1]=delta_latter
        for i in range(self.num_layers-1,1,-1):
            delta_former = np.dot(self.weights[i].T,delta_latter)*self.sigmoid_prime(self.z_history[i-1])
            self.deltas[i-1]=delta_former
            delta_latter = delta_former
    def update_params(self):
        for i in range(1,self.num_layers):
            tmp1=np.dot(self.deltas[i],self.a_history[i-1].T)
            self.weights[i] -= self.eta*tmp1/self.deltas[i].shape[1]
            self.biases[i] -= np.mean(self.eta*self.deltas[i],axis=1,keepdims=True)

    def forward(self,inputs):
        self.a_history = {}
        self.z_history = {}
        for i in range(self.num_layers-1):
            self.a_history[i]=inputs
            outputs = np.dot(self.weights[i+1],inputs)+self.biases[i+1]
            self.z_history[i+1] = outputs
            inputs = self.sigmoid(outputs)
        return outputs
    def sigmoid(self,X):
        return 1/(1+np.exp(-X))
    def sigmoid_prime(self,X):
        return self.sigmoid(X)*(1-self.sigmoid(X))
    def loss_function(self,X,y):
        return 1/2*np.sum((self.forward(X)-y)**2)/X.shape[1]
    def init_weights(self,inputs):
        weights={}
        biases={}
        for i in range(1,len(self.layers)):
            weights[i]=np.random.randn(self.layers[i],self.layers[i-1])*0.1
            biases[i]=np.zeros((self.layers[i],1))
        return weights,biases

if __name__=='__main__':
    import gzip
    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            f.read(16)
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
        return data


    def load_mnist_labels(filename):
        with gzip.open(filename, 'rb') as f:
            f.read(8)
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    def y2onehot(y):
        y_onehot = np.zeros((10,len(y)))
        idx=np.arange(len(y))
        y_onehot[y,idx] = 1
        return y_onehot
    def standardize(X):
        X_mean=np.mean(X)
        X_std=np.std(X)
        return (X-X_mean)/X_std
    x_train = load_mnist_images('../data/MNIST/train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('../data/MNIST/train-labels-idx1-ubyte.gz')
    x_test = load_mnist_images('../data/MNIST/t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('../data/MNIST/t10k-labels-idx1-ubyte.gz')
    X_train = x_train.reshape(x_train.shape[0],-1).T
    X_test = x_test.reshape( x_test.shape[0],-1).T
    y_train_onehot = y2onehot(y_train)
    X_train_standard = standardize(X_train)
    X_test_standard = standardize(X_test)
    NN = NeuralNetwork(layers=[784,28,10])
    NN.train(X_train_standard,y_train_onehot)
    plt.plot(NN.loss_history)
    plt.show()
    ans=NN.predict(X_test_standard)
    print(np.sum(ans==y_test)/len(y_test))