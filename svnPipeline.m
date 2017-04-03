% SVM
clc; clear;
%% Load dataset and generate a HOG version
%  Execute once, (if you have workspace.mat file in your path, just double click)
svmLoadSaveDataSets;

%% Train default SVM without HOG and with HOG

% Train without HOG

% divide matrix 60000 in chunks of 20000
% use one chunks to reduce execution time
%
% [xChunksTrain, chunksLabels] = divideMatrixRndInChunks(xTrain, tTrain, 3);
% select first chunk
% xChunkTrain =  xChunksTrain{1};
% chunkLabels =  chunksLabels{1};

fprintf('Compare plain with HOG to decide which data set to use\n');

SVMMdl = fitcecoc(xTrain, tTrain, 'KFold',3);
ce = kfoldLoss(SVMMdl,'LossFun','classiferror')
ce_each = kfoldLoss(SVMMdl,'LossFun','classiferror','Mode','individual')

% train with HOG features extraction
SVMMdlHOG = fitcecoc(hogTrainFeatures, tTrain, 'KFold',3);
cehog = kfoldLoss(SVMMdlHOG,'LossFun','classiferror')
ce_hog_each = kfoldLoss(SVMMdlHOG,'LossFun','classiferror','Mode','individual')

if ce < cehog
    fprintf('Low Classification error without HOG\n');
else
    fprintf('Low Classification error using HOG features\n');
end

% confirmation of the kfold results using kfold prediction
% Note: return only validation error
fprintf('Confirm result using kfoldPredict\n');
[labelsOut, score] = kfoldPredict(SVMMdl);
confMatTest = confusionmat(tTrain, labelsOut);
classAcc = sum(tTrain == labelsOut) / size(tTrain,1)
% this should throw the same value
classErr = 1 - classAcc
confAcc = 1 - (size(tTrain,1) - sum(diag(confMatTest))) / size(tTrain,1)

[labelsOutHOG, scoreHOG] = kfoldPredict(SVMMdlHOG);
confMatTestHOG = confusionmat(tTrain, labelsOutHOG);
classAccHOG = sum(tTrain == labelsOutHOG) / size(tTrain,1)
classErrHOG = 1 - classAccHOG
confAccHOG = 1 - (size(tTrain,1) - sum(diag(confMatTestHOG))) / size(tTrain,1)


%% Tune hyperparameters using HOG data

showConfusionMatrix(labelsOutHOG, hogTrainFeatures, tTrain);

%% Matlab parameter optimization
%rng default
%MdlHyperOpt = fitcecoc(hogTrainFeatures, tTrain,'OptimizeHyperparameters','auto',...
%    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%    'expected-improvement-plus'))
% comment out because it take ages!


%% use chunks to reduce execution time
fprintf('Reduce the number of training data (faster)\n');
[hogChunksTrain, hogChunksLabels] = svmDivideMatrixRndInChunks(hogTrainFeatures, tTrain, 10);

% leave orig size for final training
OrighogTrainFeatures = hogTrainFeatures;
OrigtTrain = tTrain;

hogTrainFeatures = hogChunksTrain{1};
tTrain = hogChunksLabels{1};

%% Selection of Hyperparameter optimization base in each convination
display('Selection of hyperparameters');
%pre assing space to list
hpLossResults = {};
index = 1;

for svmTemplate = 1: size(svmHpTemplates, 2)
    % train with HOG features extraction and template 
    display(strcat('Loop through templates: ', svmTemplate));   
    tempSVMMdl = fitcecoc(hogTrainFeatures, tTrain, 'KFold',3, 'Learners',svmHpTemplates{svmTemplate});
    cehog = kfoldLoss(tempSVMMdl,'LossFun','classiferror')
    ce_hog_each = kfoldLoss(tempSVMMdl,'LossFun','classiferror','Mode','individual')
    
    hpLossResults{index} = {svmTemplate, cehog, ce_hog_each};
    index = index + 1;
end

save('workspace')


%% Train on the full hog training data, Using the lowest validation error
% Loop through results and extract lower validation errors for the final
% run

%  extract the lower validation error
lowerError = 999999; % 
lowerErrorIndex = 0;
searchMatrix = {};

for i = 1:size(hpLossResults, 2)
    cehog = hpLossResults{i}{2};
    ce_hog_each = hpLossResults{i}{3};
    kernel = svmHpTemplates{i}.MakeModelInputArgs{6}; % kernel
    box = svmHpTemplates{i}.MakeModelInputArgs{8}; % box constraint
    scalar = svmHpTemplates{i}.MakeModelInputArgs{10}; % scalar

    searchMatrix{i}{1} = kernel;
    searchMatrix{i}{2} =[box scalar ce_hog_each(1) ce_hog_each(2) ce_hog_each(3) cehog];
    
    if cehog < lowerError
        lowerError = cehog;
        lowerErrorIndex = i;
    end
end

%% fit model on test data (HOG)
SVMMdl = fitcecoc(OrighogTrainFeatures, OrigtTrain, 'Learners', svmHpTemplates{lowerErrorIndex});

%predict test data
[labelsOut, score] = predict(SVMMdl, hogTestFeatures);
ConfMatTest = confusionmat(tTest, labelsOut);
Accuracy = 1 - (size(hogTestFeatures,1) - sum(diag(ConfMatTest))) / size(hogTestFeatures,1) 
Error = 1 - Accuracy

ClassAcc = sum(tTest == labelsOut) / size(tTest,1) 
ClassErr = 1 - ClassAcc

%show final confusion matrix
showConfusionMatrix(labelsOut, hogTestFeatures, tTest);

%% access svm template info
%svmHpTemplates{lowerErrorIndex}.MakeModelInputArgs{6} % kernel
%svmHpTemplates{lowerErrorIndex}.MakeModelInputArgs{8} % box constraint
%svmHpTemplates{lowerErrorIndex}.MakeModelInputArgs{10} % scalar