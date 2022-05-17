clc;
clear all;
close all;
addpath('./pca_code');

% mat files pathes
BacterialleafblightMatPath = "E:\pycharm_program\machine_learning_homework\dataset\Bacterialleafblight\BacterialHog.mat"
BrownspotMatPath = "E:\pycharm_program\machine_learning_homework\dataset\Brownspot\BrownspotHog.mat"
LeafsmutMatPath = "E:\pycharm_program\machine_learning_homework\dataset\Leafsmut\LeafsmutHog.mat"

% load data
BacterialData = load(BacterialleafblightMatPath);
BrownData = load(BrownspotMatPath);
LeafData = load(LeafsmutMatPath);
allData = [BacterialData.data;BrownData.data;LeafData.data]; % 列表示随机变量或行表示观测值的矩阵

% normalize
mean_colum = mean(allData, 1);
allDataNormalized = allData - mean_colum;
k = 2 % 降维数
[allDataNew, eigVectors, eigValues] = kPCAFunction(allDataNormalized,k,"poly",5)

% 可视化
x = allDataNew(:,1)
y = allDataNew(:,2)
scatter(x,y)

% 存储该数据集
save_path = "E:\pycharm_program\machine_learning_homework\dataset\kPCADataset.mat"
save(save_path, "allDataNew")
