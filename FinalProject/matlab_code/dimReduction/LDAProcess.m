clc;
clear all;
close all;
addpath('./lda_code');

% mat files pathes
BacterialleafblightMatPath = "E:\pycharm_program\machine_learning_homework\dataset\Bacterialleafblight\BacterialHog.mat"
BrownspotMatPath = "E:\pycharm_program\machine_learning_homework\dataset\Brownspot\BrownspotHog.mat"
LeafsmutMatPath = "E:\pycharm_program\machine_learning_homework\dataset\Leafsmut\LeafsmutHog.mat"

% load dtaa
BacterialData = load(BacterialleafblightMatPath);
BrownData = load(BrownspotMatPath);
LeafData = load(LeafsmutMatPath);
class1Data = BacterialData.data;  % num of 40
class2Data = BrownData.data; % num of 40
class3Data = LeafData.data;  % num of 40
allData = [BacterialData.data;BrownData.data;LeafData.data]; % �б�ʾ����������б�ʾ�۲�ֵ�ľ���

% ����label
label = [ones(1,40),ones(1,40)*2, ones(1,40)*3] % ����������������������

% LDA ��ά
k = 2 % ��ά
allDataNew = transpose(gda(transpose(allData), transpose(allData), label, k)) % �б�ʾ����������б�ʾ�۲�ֵ�ľ���

% ���ӻ�
x = allDataNew(:,1)
y = allDataNew(:,2)
scatter(x,y)

% �洢�����ݼ�
save_path = "E:\pycharm_program\machine_learning_homework\dataset\LDADataset.mat"
save(save_path, "allDataNew")

