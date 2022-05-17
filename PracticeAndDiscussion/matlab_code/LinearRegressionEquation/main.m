clc;
close all;
clear all;

w_values = linear_regression("sample_data.txt", 1, 0)

data = load("sample_data.txt")

% »­Í¼
figure;
sz = 10
scatter(data(:,1),data(:,2),sz,'filled')

hold on
plot(data(:,1), w_values(1,1)+data(:,1) * w_values(2,1))
