clc;
clear all;
close all;

% ��ȡ����
%data_dir = "E:\pycharm_program\machine_learning_homework\dataset\PCADataset.mat";
%data_dir = "E:\pycharm_program\machine_learning_homework\dataset\kPCADataset.mat";
%data_dir = "E:\pycharm_program\machine_learning_homework\dataset\LDADataset.mat";
%data_dir = "E:\pycharm_program\machine_learning_homework\dataset\kLDADataset.mat";
data_dir = "E:\pycharm_program\machine_learning_homework\dataset\OriginDataset.mat";
data_struct = load(data_dir);
data = data_struct.allDataNew;

% FCM����
c = 3; % �۳�3��
[centers,U,objFunc] = fcm(data,3)

% �ҵ���������
[probs, labels] = max(U)

% Ŀ�꺯�����ӻ�
figure(1);
plot(objFunc)
title("Objective Function Values")
xlabel("Iteration Count")
ylabel("Objective Function Value")

% ���������ӻ�
%index1 = find(U(1,:) == probs)
%index2 = find(U(2,:) == probs)
%index3 = find(U(3,:) == probs)
figure(2);
gscatter(transpose(data(:,1)),transpose(data(:,2)),labels)

% ���ྫ��(���������⣬Ҳ���Ǿ��������������ʵ���˳��һ��һ��)
groundtruth = [ones(1,40),ones(1,40)*2, ones(1,40)*3]
result = [groundtruth == labels]
accurate = sum(result) / 120
class1_accurate = sum(result(1:40)) / 40
class2_accurate = sum(result(41:80)) / 40
class3_accurate = sum(result(81:120)) / 40



