%% Selection of Hyperparameter by discarding the higher error rates in
% the following order: KernelFunction: linear, BoxConstrians ?

[trainChunks, labelChunks] = svmDivideMatrixRndInChunks(hogTrainFeatures, tTrain, 3);


for i=1:size(boxConstraints,2)
    
    currentTemplate = templateSVM('Standardize',1,...
                    'KernelFunction','linear',...
                    'ClassNames',{0,1,2,3,4,5,6,7,8,9}, ...
                    'BoxConstraint',boxConstraints(i))
    [linearTrainCE, linearValCE] = svmKfoldValidation(trainChunks, labelChunks, currentTemplate);
    
    trainCE(i) = mean(linearTrainCE);
    evalCE(i) = mean(linearValCE);
end

save('hog_linear_box_errors', 'trainCE', 'evalCE');