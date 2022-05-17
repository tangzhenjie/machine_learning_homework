clc;
clear all;
close all;
data = load('sample_data.txt'); % read comma separated data
x = data(:, 1);   
yT = data(:, 2);  
y = yT';
m = length(y); % number of training examples
figure(1) %ͼ1
plot(x,y, 'rx', 'MarkerSize', 10); % Plot the data
xlabel('x'); 
ylabel('y'); 
grid on;
hold on;

% ��1
x = [ones(1,m); x']; % �ӵ�һ��Ϊȫ1��֮��Ϊx1��x2
theta = zeros(2, 1); % initialize fitting parameters

% ��ʼ������
iterations = 10000; %����������
alpha = 0.01; %ѧϰ��  %�ı�ѧϰ�ʣ������һ��
s = zeros(iterations, 1);  %���ۺ����е��ۼ�ֵ
J = zeros(iterations, 1);  %���ۺ���ֵ

% �ݶ��Ż�
for k = 1:1:iterations 
    p = zeros(2, 1);  %����һ�Σ��ۼ�����
    for i = 1:1:m
        s(k) = s(k)+(theta.'*x(:,i)-y(:,i)).^2; %��J�������ۼ�
        %��ƫ��
        p = p+(theta.'*x(:,i)-y(:,i))*x(:,i); %��theta��ƫ�����ۼ�   
    end         
    J(k) = s(k)/(2*m);  %���ۺ���
    theta = theta-(alpha/m)*p;  %����theta����
    if k>1  %Ϊ������k-1������
        if J(k-1)-J(k)<1e-10   %�����С��10^2����ֹͣ����         
             break;
        end
    end
end
plot(x(2,:), theta.'* x)

figure(2)  %ͼ2
plot(J)  %�������ۺ���
ylabel('J(��)'); % Set the y axis label
xlabel('iterations'); % Set the x axis label
grid on
theta

