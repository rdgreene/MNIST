%% Selection of Hyperparameter by discarding the higher error rates in
% the following order: KernelFunction: linear, BoxConstrians 1 Scalar : ?

[trainChunks, labelChunks] = svmDivideMatrixRndInChunks(hogTrainFeatures, tTrain, 3);

boxContraint = 0.5; % Options [1     2     4     5     7    10]

for i=1:size(kernelScales,2)
    
    currentTemplate = templateSVM('Standardize',1,...
                    'KernelFunction','linear',...
                    'ClassNames',{0,1,2,3,4,5,6,7,8,9}, ...
                    'BoxConstraint',boxContraint,...
                    'KernelScale',kernelScales(i))
    [linearTrainCE, linearValCE] = svmKfoldValidation(trainChunks, labelChunks, currentTemplate);
    
    trainCE(i) = mean(linearTrainCE);
    evalCE(i) = mean(linearValCE);
end
% make sure you change name 'box1' based on boxContraint value
save('svm_hog_linear_box1_scale_errors', 'trainCE', 'evalCE');