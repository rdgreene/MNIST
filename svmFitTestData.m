%% Run Selected of Hyperparameter on Test data

[trainChunks, labelChunks] = svmDivideMatrixRndInChunks(hogTrainFeatures, tTrain, 3);

%% template with selected hyperparameters 
currentTemplate = templateSVM('Standardize',1,...
                    'KernelFunction','linear',...
                    'ClassNames',{0,1,2,3,4,5,6,7,8,9}, ...
                    'BoxConstraint',0.05,...
                    'KernelScale',0.9)


%% fit model on test data (HOG)
SVMMdl = fitcecoc(hogTrainFeatures, tTrain, 'Learners', currentTemplate);


%predict training data
[tr_labelsOut, tr_score] = predict(SVMMdl, hogTrainFeatures);
tr_ConfMatTest = confusionmat(tTrain, tr_labelsOut);
tr_Accuracy = 1 - (size(hogTrainFeatures,1) - sum(diag(tr_ConfMatTest))) / size(hogTrainFeatures,1) 
tr_Error = 1 - tr_Accuracy

tr_ClassAcc = sum(tTrain == tr_labelsOut) / size(tTrain,1) 
tr_ClassErr = 1 - tr_ClassAcc
save('svm_fit_train_data', 'tr_labelsOut', 'tr_score', 'tr_ClassAcc','tr_ClassErr');

%predict test data
[labelsOut, score] = predict(SVMMdl, hogTestFeatures);
ConfMatTest = confusionmat(tTest, labelsOut);
Accuracy = 1 - (size(hogTestFeatures,1) - sum(diag(ConfMatTest))) / size(hogTestFeatures,1) 
Error = 1 - Accuracy

ClassAcc = sum(tTest == labelsOut) / size(tTest,1) 
ClassErr = 1 - ClassAcc

showConfusionMatrix(labelsOut, hogTestFeatures, tTest);

save('svm_fit_test_data', 'labelsOut', 'score', 'ClassAcc','ClassErr');