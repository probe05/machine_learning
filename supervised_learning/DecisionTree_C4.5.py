import numpy as np
def createTree(dataSet,featureNames):
    labels = [example[-1] for example in dataSet]
    if len(dataSet[0])==1:
        return majorCnt(dataSet)
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    bestFeatureIndex = chooseBestFeatureIndex(dataSet)
    if bestFeatureIndex == -1:
        return majorCnt(dataSet)
    bestFeatureName = featureNames[bestFeatureIndex]
    del featureNames[bestFeatureIndex]
    myTree = {bestFeatureName:{}}
    uniqueFeatureValues = set([example[bestFeatureIndex] for example in dataSet])
    for featureValue in uniqueFeatureValues:
        subFeatureNames = featureNames[:]
        subDataSet = splitDataSet(dataSet,bestFeatureIndex,featureValue)
        myTree[bestFeatureName][featureValue] = createTree(subDataSet,subFeatureNames)
    return myTree
def majorCnt(dataSet):
    labels = [example[-1] for example in dataSet]
    countDict={}
    for label in labels:
        if label not in countDict.keys():
            countDict[label]=0
        countDict[label]+=1
    sortedCountDict=sorted(countDict.items(),key=lambda x:x[1],reverse=True)
    return sortedCountDict[0][0]
def chooseBestFeatureIndex(dataSet):
    baseShannonEntropy=calculateShannonEntropy(dataSet)
    bestInfoGainRatio=0
    bestFeatureIndex=-1
    for i in range(len(dataSet[0])-1):
        featureValueList = [example[i] for example in dataSet]
        uniqueFeatureValues=set(featureValueList)
        numsDataSet=len(dataSet)
        conditionalEntropy=0
        for featureValue in uniqueFeatureValues:
            subDataSet = splitDataSet(dataSet,i,featureValue)
            prob=float(len(subDataSet)/numsDataSet)
            conditionalEntropy += prob*calculateShannonEntropy(subDataSet)
        splitInfo = calculateShannonEntropy(dataSet,i)
        if splitInfo == 0:
            continue
        infoGainRatio=(baseShannonEntropy-conditionalEntropy)/splitInfo
        if infoGainRatio > bestInfoGainRatio:
            bestInfoGainRatio=infoGainRatio
            bestFeatureIndex=i
    return bestFeatureIndex
def calculateShannonEntropy(dataSet,featureIndex=-1):
    countDict={}
    targetCol=[example[featureIndex] for example in dataSet]
    numscol=len(targetCol)
    for col in targetCol:
        if col not in countDict:
            countDict[col]=0
        countDict[col]+=1
    shannonEntropy=0
    for col,count in countDict.items():
        prob=float(count/numscol)
        shannonEntropy -= prob*np.log2(prob)
    return shannonEntropy
def splitDataSet(dataSet,axis,featureValue):
    subDataSet=[]
    for example in dataSet:
        if example[axis] == featureValue:
            subexample=example[:axis]
            subexample.extend(example[axis+1:])
            subDataSet.append(subexample)
    return subDataSet
def classify(tree,featureNames,testSample):
    firstFeatureName =list(tree.keys())[0]
    secondDict = tree[firstFeatureName]
    index = featureNames.index(firstFeatureName)
    for key in secondDict.keys():
        if testSample[index] == key:
            if type(secondDict[key]).__name__ == 'dict':
                class_label = classify(secondDict[key], featureNames, testSample)
            else:
                class_label = secondDict[key]
    return class_label

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
    tree = createTree(dataSet, featureNames[:])
    print("构建的C4.5决策树：")
    print(tree)


    testSample = ['晴', '凉', '高', '弱']
    predLabel = classify(tree, featureNames, testSample)
    print("\n测试样本{}的预测结果：{}".format(testSample, predLabel))