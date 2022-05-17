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
allData = [BacterialData.data;BrownData.data;LeafData.data]; % �б�ʾ����������б�ʾ�۲�ֵ�ľ���

% normalize
mean_colum = mean(allData, 1);
allDataNormalized = allData - mean_colum;
k = 2 % ��ά��
[allDataNew, eigVectors, eigValues] = kPCAFunction(allDataNormalized,k,"poly",5)

% ���ӻ�
x = allDataNew(:,1)
y = allDataNew(:,2)
scatter(x,y)

% �洢�����ݼ�
save_path = "E:\pycharm_program\machine_learning_homework\dataset\kPCADataset.mat"
save(save_path, "allDataNew")
