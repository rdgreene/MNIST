%% Run Selected of Hyperparameter on Test data

[trainChunks, labelChunks] = svmDivideMatrixRndInChunks(hogTrainFeatures, tTrain, 3);

%% template with selected hyperparameters 
currentTemplate = templateSVM('Standardize',1,...
                    'KernelFunction','linear',...
                    'ClassNames',{0,1,2,3,4,5,6,7,8,9}, ...
                    'BoxConstraint',1,...
                    'KernelScale',0.5)


%% fit model on test data (HOG)
SVMMdl = fitcecoc(hogTrainFeatures, tTrain, 'Learners', currentTemplate);

%predict test data
[labelsOut, score] = predict(SVMMdl, hogTestFeatures);
ConfMatTest = confusionmat(tTest, labelsOut);
Accuracy = 1 - (size(hogTestFeatures,1) - sum(diag(ConfMatTest))) / size(hogTestFeatures,1) 
Error = 1 - Accuracy

ClassAcc = sum(tTest == labelsOut) / size(tTest,1) 
ClassErr = 1 - ClassAcc

showConfusionMatrix(labelsOut, hogTestFeatures, tTest);
