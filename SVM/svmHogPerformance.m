% nn coursework 1 SVM
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
fprintf('Confirm result using kfoldPredict\n');
[labelsOut, score] = kfoldPredict(SVMMdl);
confMatTest = confusionmat(tTrain, labelsOut);
classAcc = sum(tTrain == labelsOut) / size(tTrain,1)
classErr = 1 - classAcc
confAcc = 1 - (size(tTrain,1) - sum(diag(confMatTest))) / size(tTrain,1)

[labelsOutHOG, scoreHOG] = kfoldPredict(SVMMdlHOG);
confMatTestHOG = confusionmat(tTrain, labelsOutHOG);
classAccHOG = sum(tTrain == labelsOutHOG) / size(tTrain,1)
classErrHOG = 1 - classAccHOG
confAccHOG = 1 - (size(tTrain,1) - sum(diag(confMatTestHOG))) / size(tTrain,1)


showConfusionMatrix(labelsOutHOG, hogTrainFeatures, tTrain);

