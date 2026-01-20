import numpy as np
import pandas as pd

# ===================== 核心工具函数 =====================
def calc_shannon_ent(dataset):
    """
    计算数据集的香农熵（总熵）
    :param dataset: 数据集，最后一列是标签
    :return: 香农熵
    """
    # 样本总数
    num_samples = len(dataset)
    # 统计每个标签的出现次数
    label_count = {}
    for sample in dataset:
        current_label = sample[-1]  # 最后一列是标签
        if current_label not in label_count:
            label_count[current_label] = 0
        label_count[current_label] += 1

    # 计算香农熵：H = -Σ(p_i * log2(p_i))
    shannon_ent = 0.0
    for key in label_count:
        prob = float(label_count[key]) / num_samples
        shannon_ent -= prob * np.log2(prob)  # np.log2更简洁，新手易理解
    return shannon_ent


def split_dataset(dataset, axis, value):
    """
    按指定特征(axis)和特征值(value)划分数据集
    :param dataset: 原始数据集
    :param axis: 要划分的特征索引（第几个特征）
    :param value: 该特征的目标取值
    :return: 划分后的新数据集（移除了该特征列）
    """
    ret_dataset = []
    for sample in dataset:
        if sample[axis] == value:
            # 移除当前特征列，保留其他特征和标签
            reduced_sample = sample[:axis] + sample[axis + 1:]
            ret_dataset.append(reduced_sample)
    return ret_dataset


def choose_best_feature_to_split(dataset):
    """
    选择信息增益最大的特征（ID3核心）
    :param dataset: 数据集
    :return: 最优特征的索引
    """
    # 特征数量（最后一列是标签，所以减1）
    num_features = len(dataset[0]) - 1
    # 原始总熵
    base_entropy = calc_shannon_ent(dataset)
    # 初始化最优信息增益和最优特征索引
    best_info_gain = 0.0
    best_feature = -1

    # 遍历每个特征
    for i in range(num_features):
        # 提取该特征的所有取值（去重）
        feature_vals = [sample[i] for sample in dataset]
        unique_vals = set(feature_vals)  # 去重，避免重复计算

        # 计算该特征的条件熵 H(Y|X)
        new_entropy = 0.0
        for value in unique_vals:
            # 按该特征取值划分数据集
            sub_dataset = split_dataset(dataset, i, value)
            # 计算该子集的权重（样本数占比）
            prob = len(sub_dataset) / float(len(dataset))
            # 加权求和得到条件熵
            new_entropy += prob * calc_shannon_ent(sub_dataset)

        # 计算信息增益：IG = 总熵 - 条件熵
        info_gain = base_entropy - new_entropy

        # 更新最优特征
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    """
    当无特征可分时，返回出现次数最多的标签（投票法）
    :param class_list: 标签列表
    :return: 最常见的标签
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count:
            class_count[vote] = 0
        class_count[vote] += 1
    # 按出现次数降序排序，返回第一个（最多的）
    sorted_class_count = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_class_count[0][0]


# ===================== 构建ID3决策树 =====================
def create_tree(dataset, feature_names):
    """
    递归构建ID3决策树
    :param dataset: 数据集
    :param feature_names: 特征名列表（用于可视化树，方便理解）
    :return: 决策树（字典形式）
    """
    # 提取所有标签
    class_list = [sample[-1] for sample in dataset]

    # 终止条件1：所有样本标签相同（纯节点）
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # 终止条件2：无特征可分（只剩标签列）
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)

    # 选择最优分裂特征
    best_feature_idx = choose_best_feature_to_split(dataset)
    best_feature_name = feature_names[best_feature_idx]

    # 初始化决策树（字典结构：{特征名: {取值1: 子树, 取值2: 子树,...}}）
    my_tree = {best_feature_name: {}}

    # 移除已选特征名（避免递归时重复）
    del (feature_names[best_feature_idx])

    # 提取最优特征的所有取值（去重）
    feature_vals = [sample[best_feature_idx] for sample in dataset]
    unique_vals = set(feature_vals)

    # 递归构建子树
    for value in unique_vals:
        # 特征名列表的副本（避免递归时修改原列表）
        sub_feature_names = feature_names[:]
        # 递归构建子树，并添加到当前节点
        my_tree[best_feature_name][value] = create_tree(
            split_dataset(dataset, best_feature_idx, value),
            sub_feature_names
        )
    return my_tree


# ===================== 预测函数 =====================
def classify(input_tree, feature_names, test_sample):
    """
    用构建好的决策树预测单个样本
    :param input_tree: 决策树（字典）
    :param feature_names: 特征名列表
    :param test_sample: 待预测样本（无标签）
    :return: 预测标签
    """
    # 取根节点的特征名（字典的第一个key）
    first_str = list(input_tree.keys())[0]
    # 根节点的子节点（特征值→子树/标签）
    second_dict = input_tree[first_str]
    # 找到该特征名对应的索引
    feature_idx = feature_names.index(first_str)

    # 遍历该特征的取值
    for key in second_dict.keys():
        if test_sample[feature_idx] == key:
            # 如果子节点是字典（还有子树），递归预测
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feature_names, test_sample)
            # 否则是标签，直接返回
            else:
                class_label = second_dict[key]
    return class_label


# ===================== 测试代码（经典打球数据集） =====================
if __name__ == '__main__':
    dataset=pd.read_csv("../data/iris.csv")
    # 经典数据集：特征[天气, 温度, 湿度, 风速]，标签[是否打球]
    dataset = [
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
    # 特征名列表（对应数据集的前4列）
    feature_names = ['天气', '温度', '湿度', '风速']

    # 构建ID3决策树
    id3_tree = create_tree(dataset, feature_names.copy())  # 传副本避免原列表被修改
    print("构建的ID3决策树：")
    print(id3_tree)

    # 测试预测
    test_sample = ['晴', '凉', '高', '弱']  # 待预测样本（无标签）
    feature_names_test = ['天气', '温度', '湿度', '风速']  # 特征名需和构建时一致
    pred_label = classify(id3_tree, feature_names_test, test_sample)
    print("\n测试样本{}的预测结果：{}".format(test_sample, pred_label))