clc;
clear all;
close all;

% images dirs 
imagePathes = "E:\pycharm_program\machine_learning_homework\dataset\"
BacterialPathes = imagePathes + "Bacterialleafblight\"

% read the image data
data = []
imgDir = dir([BacterialPathes + "*.jpg"])
for i = 1:length(imgDir)
    imgData = imread(BacterialPathes + imgDir(i).name);
    imgDataResized = imresize(imgData,[200,400]);
    imgDataResizedGray = rgb2gray(imgDataResized)
    imgDataResizedGray_ada = adapthisteq(imgDataResizedGray,'NumTiles',[8 8],'ClipLimit',0.01,'Distribution','uniform');
    [HogFeature,vision] = extractHOGFeatures(double(imgDataResizedGray_ada),'CellSize',[80 80])
    
    %%%%%%%%%%%%% visilation tools %%%%%%%%%%%%%%%%%
    %subplot(1,2,1);
    %imshow(imgData);
    %subplot(1,2,2);
    %imshow(imgDataResizedGray_ada);
    
    
    %f = HogFeature;
    %figure;
    %imshow(imgDataResizedGray);
    %hold on
    %plot(vision)
    %%%%%%%%%%%%% visilation tools %%%%%%%%%%%%%%%%%
    
    data(i,:) = HogFeature
end

% save the image data
save_path = BacterialPathes + "BacterialHog.mat"
save(save_path, "data")


