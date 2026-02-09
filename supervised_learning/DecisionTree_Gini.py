import numpy as np
def createTree(dataSet,featureNames):
    labels=[sample[-1] for sample in dataSet]
    if len(dataSet[0]) == 1:
        return majorCnt(dataSet)
    if len(labels) == labels.count(labels[0]):
        return labels[0]
    bestFeatureIndex,bestSubFeatureName = chooseBestFeatureIndex(dataSet)
    bestFeatureName = featureNames[bestFeatureIndex]
    subNodeName = f'{bestFeatureName}=={bestSubFeatureName}'
    myTree = {subNodeName:{}}
    leftDataSet,rightDataSet=splitDataSet(dataSet,bestFeatureIndex,bestSubFeatureName)
    myTree[subNodeName]['yes'] = createTree(leftDataSet,featureNames)
    myTree[subNodeName]['no'] = createTree(rightDataSet, featureNames)
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
    minGini = float('inf')
    bestFeatureIndex=-1
    bestSubFeatureName=''
    numSamples=len(dataSet)
    for i in range(len(dataSet[0])-1):
        uniqueSubFeatureNames = set([sample[i] for sample in dataSet])
        for j in uniqueSubFeatureNames:
            leftDataSet,rightDataSet = splitDataSet(dataSet,i,j)
            leftGini  = calcGini(leftDataSet)
            rightGini = calcGini(rightDataSet)
            ln=len(leftDataSet)
            rn=len(rightDataSet)
            Gini = ln/numSamples*leftGini + rn/numSamples*rightGini
            if Gini < minGini:
                minGini=Gini
                bestFeatureIndex=i
                bestSubFeatureName=j
    return bestFeatureIndex,bestSubFeatureName
def splitDataSet(dataSet,axis,subFeatureName):
    leftDataSet,rightDataSet=[],[]
    for sample in dataSet:
        if sample[axis]==subFeatureName:
            leftDataSet.append(sample)
        else:
            rightDataSet.append(sample)
    return leftDataSet,rightDataSet
def calcGini(dataSet,featureIndex=-1):
    targetCol = [sample[featureIndex] for sample in dataSet]
    targetCount={}
    for target in targetCol:
        if target not in targetCount:
            targetCount[target]=0
        targetCount[target]+=1
    return 1-np.sum((np.array(list(targetCount.values()))/len(targetCol))**2)


def classify(tree, featureNames, testSample):
    firstFeatureName = list(tree.keys())[0]
    secondDict = tree[firstFeatureName]
    feature_name, split_value = firstFeatureName.split('==')
    index = featureNames.index(feature_name)
    for key in secondDict.keys():
        if (testSample[index] == split_value and key == 'yes') or (testSample[index] != split_value and key == 'no'):
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
    print("构建的cart决策树：")
    print(tree)

    testSample = ['晴', '凉', '高', '弱']
    predLabel = classify(tree, featureNames, testSample)
    print("\n测试样本{}的预测结果：{}".format(testSample, predLabel))