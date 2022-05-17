clc;
clear all;
close all;
addpath('./lda_code');

% mat files pathes
BacterialleafblightMatPath = "E:\pycharm_program\machine_learning_homework\dataset\Bacterialleafblight\BacterialHog.mat"
BrownspotMatPath = "E:\pycharm_program\machine_learning_homework\dataset\Brownspot\BrownspotHog.mat"
LeafsmutMatPath = "E:\pycharm_program\machine_learning_homework\dataset\Leafsmut\LeafsmutHog.mat"

% load data
BacterialData = load(BacterialleafblightMatPath);
BrownData = load(BrownspotMatPath);
LeafData = load(LeafsmutMatPath);
class1Data = BacterialData.data;  % num of 40
class2Data = BrownData.data; % num of 40
class3Data = LeafData.data;  % num of 40
allData = [BacterialData.data;BrownData.data;LeafData.data]; % 列表示随机变量或行表示观测值的矩阵

% 生成label
label = [ones(1,40),ones(1,40)*2, ones(1,40)*3] % 样本个数根据数据来构建

% 降维维数
k = 2 

% 核函数参数
option.KernelType = "poly"
option.KernelPars = [5]

% kLDA 降维
allDataNew = transpose(gda(transpose(allData), transpose(allData), label, k,option)) % 列表示随机变量或行表示观测值的矩阵

% 可视化
x = allDataNew(:,1)
y = allDataNew(:,2)
scatter(x,y)

% 存储该数据集
save_path = "E:\pycharm_program\machine_learning_homework\dataset\kLDADataset.mat"
save(save_path, "allDataNew")

