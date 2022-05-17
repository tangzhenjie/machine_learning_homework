x1 = [0;1]
x2 = [1;0]
x3 = [1;1]
x4 = [2;1]

% ���Ժ�
options.KernelType = 'linear'
matrix1 = [
    kernel(x1,x1,options),kernel(x1,x2,options), kernel(x1,x3,options);
    kernel(x2,x1,options),kernel(x2,x2,options), kernel(x2,x3,options);
    kernel(x3,x1,options),kernel(x3,x2,options), kernel(x3,x3,options);
    ]

% ����ʽ��
options.KernelType = 'poly'
options.KernelPars = 2
matrix2 = [
    kernel(x1,x1,options),kernel(x1,x2,options), kernel(x1,x3,options);
    kernel(x2,x1,options),kernel(x2,x2,options), kernel(x2,x3,options);
    kernel(x3,x1,options),kernel(x3,x2,options), kernel(x3,x3,options);
    ]
% ��˹��
options.KernelType = 'rbf'
options.KernelPars = 5
matrix3 = [
    kernel(x1,x1,options),kernel(x1,x2,options), kernel(x1,x3,options);
    kernel(x2,x1,options),kernel(x2,x2,options), kernel(x2,x3,options);
    kernel(x3,x1,options),kernel(x3,x2,options), kernel(x3,x3,options);
    ]