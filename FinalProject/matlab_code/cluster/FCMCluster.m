clc;
clear all;
close all;

% 读取数据
%data_dir = "E:\pycharm_program\machine_learning_homework\dataset\PCADataset.mat";
%data_dir = "E:\pycharm_program\machine_learning_homework\dataset\kPCADataset.mat";
%data_dir = "E:\pycharm_program\machine_learning_homework\dataset\LDADataset.mat";
%data_dir = "E:\pycharm_program\machine_learning_homework\dataset\kLDADataset.mat";
data_dir = "E:\pycharm_program\machine_learning_homework\dataset\OriginDataset.mat";
data_struct = load(data_dir);
data = data_struct.allDataNew;

% FCM聚类
c = 3; % 聚成3类
[centers,U,objFunc] = fcm(data,3)

% 找到所属的类
[probs, labels] = max(U)

% 目标函数可视化
figure(1);
plot(objFunc)
title("Objective Function Values")
xlabel("Iteration Count")
ylabel("Objective Function Value")

% 聚类结果可视化
%index1 = find(U(1,:) == probs)
%index2 = find(U(2,:) == probs)
%index3 = find(U(3,:) == probs)
figure(2);
gscatter(transpose(data(:,1)),transpose(data(:,2)),labels)

% 聚类精度(计算有问题，也就是聚类出来的类别和真实类别顺序不一定一样)
groundtruth = [ones(1,40),ones(1,40)*2, ones(1,40)*3]
result = [groundtruth == labels]
accurate = sum(result) / 120
class1_accurate = sum(result(1:40)) / 40
class2_accurate = sum(result(41:80)) / 40
class3_accurate = sum(result(81:120)) / 40



