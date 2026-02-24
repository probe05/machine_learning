import numpy as np
class NaiveBayesClassifier():
    def __init__(self):
        self.classes=None
        self.priorProbs = {}
        self.conditionalProbs = {}
    def _calcPriorP(self,y):
        for cls in self.classes:
            self.priorProbs[cls] = np.sum([cls==yc for yc in y])/len(y)
    def _calcConditionalP(self,X,y):
        for cls in self.classes:
            self.conditionalProbs[cls] = {}
            for j in range(len(X[0])):
                subX = [x[j] for x in X]
                subFeatureUnique=list(set(subX))
                self.conditionalProbs[cls][j]={}
                for subFeature in subFeatureUnique:
                    a=[cls==yc for yc in y]
                    b=[subFeature==subXc for subXc in subX]
                    c=[a[i] and b[i] for i in range(len(a))]
                    self.conditionalProbs[cls][j][subFeature] = (np.sum(c)+1)/(np.sum([cls==yc for yc in y])+len(subFeatureUnique))
    def fit(self,X,y):
        self.classes = list(set(y))
        self._calcPriorP(y)
        self._calcConditionalP(X,y)
    def predict(self,X):
        probs={}
        i=0
        for cls in self.classes:
            prob = self.priorProbs[cls]
            for j in range(len(X)):
                prob *= self.conditionalProbs[cls][j][X[j]]
            probs[cls] = prob
        print(probs)
        return sorted(probs.items(), key=lambda x:x[1], reverse=True)[0][0]
if __name__ == '__main__':
    dataSet = [
        ['晴', '热', '高', '弱', '否'],
        ['晴', '热', '高', '强', '否'],
        ['阴', '热', '高', '弱', '是'],
        ['雨', '中', '高', '弱', '是'],
        ['雨', '凉', '正常', '弱', '是'],
        ['雨', '凉', '正常', '强', '否'],
        ['阴', '凉', '正常', '强', '是'],
        ['晴', '中', '高', '弱', '否'],
        ['晴', '凉', '正常', '弱', '是'],
        ['雨', '中', '正常', '弱', '是'],
        ['晴', '中', '正常', '强', '是'],
        ['阴', '中', '高', '强', '是'],
        ['阴', '热', '正常', '弱', '是'],
        ['雨', '中', '高', '强', '否']
    ]
    featureNames = ['天气', '温度', '湿度', '风速']
    X = [data[:-1] for data in dataSet]
    y = [data[-1] for data in dataSet]
    bayesClassifier = NaiveBayesClassifier()
    bayesClassifier.fit(X,y)
    print(bayesClassifier.conditionalProbs)
    testSample = ['晴', '凉', '高', '弱']
    predLabel = bayesClassifier.predict(testSample)
    print("\n测试样本{}的预测结果：{}".format(testSample, predLabel))