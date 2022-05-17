clc;
clear all;
close all;

% images dirs 
imagePathes = "E:\pycharm_program\machine_learning_homework\dataset\"
BacterialPathes = imagePathes + "Bacterialleafblight\"

% Open a pool of MATLAB workers
poolobj = parpool(5);


% read the image data
data1 = []
imgDir = dir([BacterialPathes + "*.jpg"])
parfor i = 1:length(imgDir)
    imgData = imread(BacterialPathes + imgDir(i).name);
    imgDataResized = imresize(imgData,[200,400]);
    imgDataResizedData = reshape(imgDataResized, [1, 200*400*3])
    data1(i,:) = imgDataResizedData
end

data2 = []
BrownspotPathes = imagePathes + "Brownspot\"
imgDir = dir([BrownspotPathes + "*.jpg"])
parfor i = 1:length(imgDir)
    imgData = imread(BrownspotPathes + imgDir(i).name);
    imgDataResized = imresize(imgData,[200,400]);
    imgDataResizedData = reshape(imgDataResized, [1, 200*400*3])
    data2(i,:) = imgDataResizedData
end

data3 = []
LeafsmutPathes = imagePathes + "Leafsmut\"
imgDir = dir([LeafsmutPathes + "*.jpg"])
parfor i = 1:length(imgDir)
    imgData = imread(LeafsmutPathes + imgDir(i).name);
    imgDataResized = imresize(imgData,[200,400]);
    imgDataResizedData = reshape(imgDataResized, [1, 200*400*3])
    data3(i,:) = imgDataResizedData
end

% Stop MATLAB workers
delete(poolobj);

allDataNew = double([data1;data2;data3])
% 存储该数据集
save_path = "E:\pycharm_program\machine_learning_homework\dataset\OriginDataset.mat"
save(save_path, "allDataNew")









