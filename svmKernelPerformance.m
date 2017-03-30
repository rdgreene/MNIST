%% Selection of Hyperparameter by discarding the higher error rates in
% the following order: KernelFunction, BoxConstrians and Scalar value

[trainChunks, labelChunks] = svmDivideMatrixRndInChunks(hogTrainFeatures, tTrain, 3);

% Select best performing Kernel (from linear, polynomial and gaussian)
[linearTrainCE, linearValCE] = svmKfoldValidation(trainChunks, labelChunks, linearTemplate);
[gaussianTrainCE, gaussianValCE] = svmKfoldValidation(trainChunks, labelChunks, gaussianTemplate);
[polynomialTrainCE, polynomialValCE] = svmKfoldValidation(trainChunks, labelChunks, polynomialTemplate);


linearValMean = mean(linearValCE);
gaussianValMean = mean(gaussianValCE);
polynomialValMean = mean(polynomialValCE);


if linearValMean < gaussianValMean && linearValMean < polynomialValMean
    fprintf('linear function has the best performance %d', linearValMean);
end

if gaussianValMean < linearValMean && gaussianValMean < polynomialValMean
    fprintf('gaussian function has the best performance %d', gaussianValMean);
end

if polynomialValMean < linearValMean && polynomialValMean < gaussianValMean
    fprintf('polynomial function has the best performance %d', polynomialValMean);
end

save('svm_kernel_error','linearValMean', 'gaussianValMean', 'polynomialValMean');