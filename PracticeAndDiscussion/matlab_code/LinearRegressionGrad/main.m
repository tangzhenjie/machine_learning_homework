clc;
clear all;
close all;
data = load('sample_data.txt'); % read comma separated data
x = data(:, 1);   
yT = data(:, 2);  
y = yT';
m = length(y); % number of training examples
figure(1) %图1
plot(x,y, 'rx', 'MarkerSize', 10); % Plot the data
xlabel('x'); 
ylabel('y'); 
grid on;
hold on;

% 补1
x = [ones(1,m); x']; % 加第一列为全1，之后为x1、x2
theta = zeros(2, 1); % initialize fitting parameters

% 初始化参数
iterations = 10000; %迭代最大次数
alpha = 0.01; %学习率  %改变学习率，结果不一样
s = zeros(iterations, 1);  %代价函数中的累加值
J = zeros(iterations, 1);  %代价函数值

% 梯度优化
for k = 1:1:iterations 
    p = zeros(2, 1);  %迭代一次，累计清零
    for i = 1:1:m
        s(k) = s(k)+(theta.'*x(:,i)-y(:,i)).^2; %求J函数的累加
        %求偏导
        p = p+(theta.'*x(:,i)-y(:,i))*x(:,i); %对theta求偏导的累加   
    end         
    J(k) = s(k)/(2*m);  %代价函数
    theta = theta-(alpha/m)*p;  %更新theta参数
    if k>1  %为了下面k-1有索引
        if J(k-1)-J(k)<1e-10   %若误差小于10^2，则停止迭代         
             break;
        end
    end
end
plot(x(2,:), theta.'* x)

figure(2)  %图2
plot(J)  %画出代价函数
ylabel('J(θ)'); % Set the y axis label
xlabel('iterations'); % Set the x axis label
grid on
theta

