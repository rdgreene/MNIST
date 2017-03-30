function [ kFoldTrainCE, kFoldValCE ] = kFoldValidateNN( folds, trainingData, trainingLabels, tTemplate )
% Use input data and parameters to run k-fold cross validation and return
% error metrics

    %{
    INPUTS:
    folds - number of foldes to use for cross validation
    trainingData - matrix data for training neural net, [d1 = features, d2 = examples] [input parameter for netTrain]
    trainingLabels - vector of training labels corresponding to each training example in trainingData [input parameter for netTrain]

    OUTPUTS:
    kFoldClassErrors - matrix of training errors (classification) for each epoch in each fold [d1 = epoch, d2 = fold] 
    kFoldVectorErrors - matrix of training errors (eucdlidian distance) for each epoch in each fold [d1 = epoch, d2 = fold] 
    %}

    fprintf('\n\n***RUNNING %d-FOLD CROSS VALIDATION***\n', folds)

    % modifier for rng seed (ensures dataset is shuffled differently in each fold)
    rngModifier = 36;
    
    % create target vectors for calculating error when training
    %targetVectors = createMnistTargetVectors(trainingLabels);
    
    % calculate number of examples to use for training each fold
    examplesPerFold = size(trainingData, 2)/folds;
    
    % create indices for partitioning dataset into k-folds
    for k = 1:folds
        kFoldIdx(k, 1) = 1+(examplesPerFold*(k-1));
        kFoldIdx(k, 2) = examplesPerFold*k;
    end

    % train and validate folds
    for k = 1:folds

        tic    

        fprintf('\nFOLD %d ', k)

        % partition training and validation data for current iteration 
        foldTrainData = trainingData;
        foldTrainData(:, kFoldIdx(k, 1):kFoldIdx(k, 2)) = [];
        foldValData = trainingData(:, kFoldIdx(k, 1):kFoldIdx(k, 2));

        % create target vectors for training and validation datasets for current iteration
        foldTrainLabels = trainingLabels;
        foldTrainLabels(kFoldIdx(k, 1):kFoldIdx(k, 2)) = [];
        foldValLabels = trainingLabels(kFoldIdx(k, 1):kFoldIdx(k, 2));

        % train neural net for current iteration
        mdl = fitcecoc(foldTrainData, foldTrainLabels, 'Learners', tTemplate);

        trainPreds = predict(mdl, foldTrainData); 
        trainAccuracy = trainPreds == foldTrainLabels;
        trainAccuracy = sum(trainAccuracy)/size(trainAccuracy,1);
        trainError = 1 - trainAccuracy;

        valPreds = predict(mdl, foldValData); 
        valAccuracy = valPreds == foldValLabels;
        valAccuracy = sum(valAccuracy)/size(valAccuracy,1);
        valError = 1 - valAccuracy;


        % store vector of training and validation errors for each epoch of current iteration in matrix to return as output 
        kFoldTrainCE(k) = trainError;
        kFoldValCE(k) = valError;

        fprintf('[Fold %d error: %.4f] ', [k valError])

        % increment rngModifier for training next fold
        rngModifier =  rngModifier + 1;

        toc
    
    end
    
    % print mean training and validaiton error for k-folds to console 
    fprintf('\nSVM trained and validated using %d fold(s).\n', k)
    fprintf('TRAINING RESULTS [Classification Error: %.4f.]\n', mean(kFoldTrainCE))
    fprintf('VALIDATION RESULTS [Classification Error: %.4f]\n', mean(kFoldValCE))
  
end

