function [te, ve] = svmKfoldValidation(dataChunks, labelChunks, template)

    noOfChunks = size(dataChunks,2);
    indxPermutations = randperm(noOfChunks); 
    
    for i = 1:noOfChunks
        idx = indxPermutations(i);
        chunkValidation = dataChunks{idx};
        chunkValidationLabels = labelChunks{idx};
        
        chunkTrain = [];
        chunkTrainLabels = [];
        
        for ii = 1:noOfChunks
            if ii ~= i
                tidx = indxPermutations(ii);
                chunkTrainChunk = dataChunks{tidx};
                chunkTrainLabelsChunk = labelChunks{tidx};
                chunkTrain = [chunkTrain ; chunkTrainChunk];
                chunkTrainLabels = [chunkTrainLabels; chunkTrainLabelsChunk];
            end
        end
        fprintf('-- fold %d', i); 
        % Trainning
        SVMMdl = fitcecoc(chunkTrain, chunkTrainLabels, 'Learners',template);
        [labelsOut, score] = predict(SVMMdl, chunkTrain);

        ConfMatTest = confusionmat(chunkTrainLabels, labelsOut);
        TrainingAccuracy = 1 - (size(chunkTrain,1) - sum(diag(ConfMatTest))) / size(chunkTrain,1) 
        
        size(chunkTrainLabels, 1) 
        trainClassAcc = sum(chunkTrainLabels == labelsOut) / size(chunkTrainLabels, 1) 
        trainClassErr = 1 - trainClassAcc
        
        
        % Validation
        [labelsOut, score] = predict(SVMMdl, chunkValidation);

        ConfMatTest = confusionmat(chunkValidationLabels, labelsOut);
        ValidationAccuracy = 1 - (size(chunkValidation,1) - sum(diag(ConfMatTest))) / size(chunkValidation,1) 

        valClassAcc = sum(chunkValidationLabels == labelsOut) / size(chunkValidation, 1) 
        valClassErr = 1 - valClassAcc
        te(i) = trainClassErr;
        ve(i) = valClassErr;
    end