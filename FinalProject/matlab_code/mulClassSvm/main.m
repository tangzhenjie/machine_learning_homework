% clear screen and variable space
clc; clear all; close all;

% Open a pool of MATLAB workers
poolobj = parpool(5);

% a few parameters to be set
input_layer_size = 2; % the feature dimention after using PCA/kPCA/LDA/kLDA
num_labels = 3;       % 3 classes to classify into
%data_dir = "E:\pycharm_program\machine_learning_homework\dataset\PCADataset.mat";  
%data_dir = "E:\pycharm_program\machine_learning_homework\dataset\kPCADataset.mat";
%%%%%%%%%%%%%%%%%%%%%%%%%%% some logical problem for LDA data %%%%%%%%%%%%%%%%%%%
%data_dir = "E:\pycharm_program\machine_learning_homework\dataset\LDADataset.mat";
%data_dir = "E:\pycharm_program\machine_learning_homework\dataset\kLDADataset.mat";
data_dir = "E:\pycharm_program\machine_learning_homework\dataset\OriginDataset.mat";

% =========== Load the whole data ========================
fprintf('Loading the whole data...\n');
data_struct = load(data_dir);
X = data_struct.allDataNew;
y = transpose([ones(1,40),ones(1,40)*2, ones(1,40)*3]) % the labels of this dataset
m = size(X, 1); % number of training samples


fprintf('Program paused. Press ENTER to continue...\n');
pause;

% =============== Shuffle the whole dataset =============================
%rand_indices = randperm(m); % comment it 
load('rand_indices.mat')    % fix the rand_indices
X = X(rand_indices, :);
y = y(rand_indices, :);

% =============== Normalize features in data ===========================
[X_norm, mu, sigma] = featureNormalize(X);

% =============== Convert y into form for 1-vs-all =====================
y_new = makeClasses(y, num_labels);

% =============== Split dataset into training and test sets ===========
% make a 80%, 20% split
split = m * 0.8;
% training set
X_train = X_norm(1:split, :);
y_train = y_new(1:split, :);

% test set
X_test = X_norm(split+1:end, :);
y_test = y_new(split+1:end, :);

% =============== Initialise an array of svms (structure) =============
% there are num_labels svm's to consider
svm_array(num_labels) = struct('X', [], 'y', [], 'kernelFunction', 'linearKernel', 'b', [], 'alphas', [], 'w', []);
C_svm = 1; %  sigma_svm = 0.1; additional SVM parameters


% ============== Train SVMs ===========================================
parfor i = 1: num_labels
    fprintf('\nChoosing parameters and training SVMs. This might take a long time ....\n');
    % training X = X_train
    % training Y = y_train(:, i)
    fprintf('Training SVM for detecting class: %d\n', i);
    % Do uncomment the line below if time is not an issue
    % Performing parameter selection automatically should vastly improve accuracy
    %[C_svm, sigma_svm] = dataset3Params(X_train, y_train(:, i), X_cv, y_cv(:, i));
    svm_array(i) = svmTrain(X_train, y_train(:, i), C_svm, @linearKernel);
end

fprintf('\nTraining complete... Press ENTER to continue...\n');
save('svmArray.mat', 'svm_array');
% Stop MATLAB workers
delete(poolobj);

% ============= Predict for all SVMs =================================
% Use cross validation set
fprintf('\n===============\nRunning predictions on test set and deriving accuracy');
count = 0;  % number of correctly predicted terms
for i = 1:size(X_test, 1)
    input = X_test(i, :);
    expected = y_test(i, :);
    % convert expected into a single number from the classes array
    expectedY = 0;
    for j = 1:num_labels
       if expected(j) == 1 
           expectedY = j;
           break;
       end
    end
    % check the expectedY SVM's output
    prediction = svmPredict(svm_array(expectedY), input);
    if prediction == 1
        count = count + 1;
    end
end
success = 100 * count/size(X_test, 1);
format long g;
fprintf('\nTesting accuracy: %d %\n', success);
format short;

