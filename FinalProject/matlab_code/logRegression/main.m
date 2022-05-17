% clear screen and variable space
clc; clear all; close all;


% dataset dir
%data_dir = "E:\pycharm_program\machine_learning_homework\dataset\PCADataset.mat";  
%data_dir = "E:\pycharm_program\machine_learning_homework\dataset\kPCADataset.mat";
%%%%%%%%%%%%%%%%%%%%%%%%%%% some logical problem for LDA data %%%%%%%%%%%%%%%%%%%
%data_dir = "E:\pycharm_program\machine_learning_homework\dataset\LDADataset.mat";
%data_dir = "E:\pycharm_program\machine_learning_homework\dataset\kLDADataset.mat";
data_dir = "E:\pycharm_program\machine_learning_homework\dataset\OriginDataset.mat";

% =======================load dataset=============================%
data_struct = load(data_dir);
data = data_struct.allDataNew;
label = transpose([ones(1,40),ones(1,40)*2, ones(1,40)*3]) % the labels of this dataset
dataset = [data,label]

% ================= Shuffle the whole dataset=====================%
m = size(dataset, 1); % number of dataset
%rand_indices = randperm(m); % comment it due to using the same train/test
load('rand_indices.mat')    % fix the rand_indices
dataset = dataset(rand_indices, :);


% =============== Split dataset into training and test sets ===========%
% make a 80%, 20% split
split = m * 0.8;
% training set
dataset_train = dataset(1:split, :);


% test set
dataset_test = dataset(split+1:end, :);


% ==================== logistic_regression ========================%
logistic_regression(dataset_train,1,dataset_test)