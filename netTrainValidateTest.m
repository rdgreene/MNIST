%{

TASK LIST
[x] add confidence metric to predicitons output
[x] implement weight decay functionality
[] check rng seeding is appropriate
[] add script for ploting confusion matrix
[] add script for plot cross-validation results
[] Examine mini-batch issues (try 'accumulating deltas')
[] add funcitonality to save network weights to file
[] change all 'alpha' to 'eta'
[] change ED and Vec references to MSE
[] rename createMnistTargetVectors to something more appropriate

%}

%% INIT
clear; clc; close all

% data mods
clip = 0; clipSize = 1000; % reduce size of training dataset to amount defined by 'clipSize' variable
binarizeTrainingData = 0; % pre-process data prior to training using binarization

% run mods
displayClassFrequencies = 0;  % run script to print table of class frequencies to console
runHyperParameterTuning = 1; % run script to tune hyper-parameters using k-fold cross validation
trainFinalNetwork = 0; % run script to train and evaluate final network
validateInFinalTrain = 0;  valPer = 0.1; % use validation partition in training final network; set percentage of data set to use for validation in final training run
plotMetrics = 0; 
k = 3; % number of folds for cross validation

%% INIT DEFUALT NETWORK ARCHITECTURE & HYPER-PARAMETERS

% number of neurons in hidden layer
hiddenLayerSize = 30;

% hyper-parameters.
alpha = 0.1; % learning rate
mu = 0; % momentum
lambda = 0; % weight decay factor
 
%epochs & batch size
epochs = 75;
batchSize = 1; % set to 1 for SGD

%% LOAD & TABULATE DATA

% load MNIST data set using Stanford's functions
trainingData = loadMNISTImages('train-images.idx3-ubyte');
trainingLabels = loadMNISTLabels('train-labels.idx1-ubyte');

if displayClassFrequencies == true

    % calculate number of examples in each class
    classes = [0:9; zeros(1,10)]';
    for c = 1:size(classes, 1)
        for i = 1:size(trainingLabels,1)        
            if trainingLabels(i) == classes(c, 1)
                classes(c, 2) = classes(c, 2) + 1; 
            end
        end
    end 

    % calculate percentage of training examples belonging to each class
    classes(:,3) = classes(:, 2) ./ sum(classes(:,2));

    % print class frequncy table to console
    fprintf('~CLASS FREQUENCIES~\n') 
    for c = 1:size(classes, 1)
        fprintf('Class %d: %.2f (%d)\n', [classes(c, 1) classes(c, 3) classes(c, 2)])
    end

    % clear redundant variables from workspace
    clear c i 

    pause(1)
    
end

%% SHUFFLE AND PROCESS TRAINING DATASET

% randomly shuffle data (rng seeding used for reproducibility)
rng(17); shuffle = randperm(size(trainingData, 2));
trainingData = trainingData(:, shuffle);
trainingLabels = trainingLabels(shuffle);

% OPTIONAL: binarize image data (pre-processing)
if binarizeTrainingData == true
   trainingData = imbinarize(trainingData);
end

% OPTIONAL: clip size of training dataset
if clip == true
    trainingData = trainingData(:, 1:clipSize); trainingLabels = trainingLabels(1:clipSize); clear clipSize
end
    
%% HYPER-PARAMETER TUNING WITH K-FOLD CROSS VALIDATION

if runHyperParameterTuning == true

    % define serach vectors for each hyper parameter
    pHiddenLayerSize = [300]%[10 50 100]; % [10 20 40 60 80 100 120 140 160 180 200 250 300 350 400];
    pAlpha = [0.01 0.03 0.1 0.3 0.9]; 
    pMu = [0.01 0.03 0.1 0.3 0.9];
    pLambda = [0.0001 0.0003 0.001 0.003 0.01];

    cnt = 0;
    
    % loop through values in hyper-parameter search vector and evaluate using k-fold crOss validation
    for s = 1:size(pHiddenLayerSize,2)
        for a = 1:size(pAlpha,2)
            
            cnt = cnt +1;

            %define hyper-parameter for for current iteration
            hiddenLayerSize = pHiddenLayerSize(s);
            alpha = pAlpha(a);

            % train & validate using k-folds and return validation metrics
            [kFoldTrainCE, kFoldTrainMSE, kFoldValCE, kFoldValMSE] = netKFoldValidate(k, hiddenLayerSize, alpha, mu, lambda, epochs, batchSize, trainingData, trainingLabels);

            % ADD: script to save final validation results for each loop (i.e. mean(kFoldClassErrors(end, :))

            sizeTrack(cnt) = hiddenLayerSize;
            alphaTrack(cnt) = alpha;
            kFoldTrainAveCE(:,cnt) = mean(kFoldTrainCE, 2);
            kFoldValAveCE(:,cnt) = mean(kFoldValCE, 2);
            kFoldTrainAveMSE(:,cnt) = mean(kFoldTrainMSE, 2);
            kFoldValAveMSE(:,cnt) = mean(kFoldValMSE, 2);

            
            
        end
    end
end

%% PLOT CROSS-VALIDATION RESULTS

if runHyperParameterTuning == true
    if plotMetrics == true

%         kFoldTrainAveCE = mean(kFoldTrainCE, 2);
%         kFoldValAveCE = mean(kFoldValCE, 2);
%         kFoldTrainAveMSE = mean(kFoldTrainMSE, 2);
%         kFoldValAveMSE = mean(kFoldValMSE, 2);

        % plot vector distance error
        subplot(2,1,1);
        hold on;
        plot(kFoldTrainAveMSE, 'k--')
        plot(kFoldValAveMSE, 'r-')
        title('Mean Squared Error: Training vs Validation'); 
        xlabel('Epoch'); ylabel('Error'); 
        legend('Training', 'Validation');

        % plot classification error
        subplot(2,1,2);
        hold on;
        plot(kFoldTrainAveCE, 'k--')
        plot(kFoldValAveCE, 'r-')
        title('Classification Error: Training vs Validation'); 
        xlabel('Epoch'); ylabel('Error'); 
        legend('Training', 'Validation');
    
    end
end

%%
% figure;
% plot(kFoldValAveCE(end,:))
% hold on
% plot(kFoldTrainAveCE(end,:), '--')

% save plot to wd
% print('validationResults','-dpng');

%% TRAIN FINAL NETWORK WITH CHOSEN PARAMETERS

if trainFinalNetwork == true

    % number of neurons in hidden layer
    hiddenLayerSize = hiddenLayerSize;

    % hyper-parameters.
    alpha = alpha; % learning rate
    mu = mu; % momentum
    lambda = lambda; % weight decay factor

    %epochs & batch size
    epochs = epochs;
    batchSize = batchSize; % set to 1 for SGD

    % if valFinal is true, then use validation fold for early stopping while training final model
    if validateInFinalTrain == true

        % create index for splitting traing dataset into training and validaiton partitions
        valIdx = round((size(trainingData, 2)*(1-valPer)));

        % create validation split for early stopping
        rng(17); shuffle = randperm(size(trainingData, 2));
        trainingData = trainingData(:, shuffle);
        trainingLabels = trainingLabels(shuffle);
        fTrainingData = trainingData(:, 1:valIdx);
        fTrainingLabels = trainingLabels(1:valIdx);
        fValData = trainingData(:, valIdx+1:end);
        fValLabels = trainingLabels(valIdx+1:end);

        % create target vectors for training data
        targetVectors = createMnistTargetVectors(fTrainingLabels);

        fprintf('\n\n***TRAINING FINAL NETWORK***\n')

        % train network using full training dataset
        [ weightsHid, weightsOut, epochAccuracies, epochErrors, epochValClassErrors, epochValMSE ] = netTrain( hiddenLayerSize, batchSize, epochs, alpha, mu, lambda, fTrainingData, targetVectors, 0, fValData, fValLabels);


    else % otherwise train final model using all data for training

        % create target vectors for training data
        targetVectors = createMnistTargetVectors(trainingLabels);

        fprintf('\n\n***TRAINING FINAL NETWORK***\n')

        % train network using full training dataset
        [ weightsHid, weightsOut, epochAccuracies, epochErrors ] = netTrain( hiddenLayerSize, batchSize, epochs, alpha, mu, lambda, trainingData, targetVectors, 0);

    end

end

%% EVALUATE TRAINED NETWORK USING TEST DATASET

if trainFinalNetwork == true

    fprintf('\nEvaluating network using test dataset.')

    % Load test data.
    testingData = loadMNISTImages('t10k-images.idx3-ubyte');
    testingLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');

    % if option selected at top, binarize image data
    if binarizeTrainingData == true
       testingData = imbinarize(testingData);
    end

    % calculate evaluation metrics for test dataset
    [testClassaccuracy, testClasserror, testVecError, testMetrics, predictions] = netEvaluate( testingData, testingLabels, weightsHid, weightsOut, true );

    % get index of incorrectly classified test examples
    misclassfiedIdx = find(~predictions(:,3));
   
    % split 'predicitons' into correctly / incorrectly classified examples
    predictionsCorrect = predictions(find(predictions(:,3)), :);
    predictionsIncorrect = predictions(misclassfiedIdx, :);
    
            
    % creat confusion matrix
    confusionMatrix = confusionmat(predictions(:,2), predictions(:,1));
    
end

%% PLOT METRICS FOR FINAL TRAINING RUN

if trainFinalNetwork == true
    if plotMetrics == true
        
        % plot confidence distributions for correct vs incorrectly classified
        figure;
        histogram(predictionsIncorrect(:,4), 20, 'Normalization','probability', 'FaceColor', 'r')
        hold on;
        histogram(predictionsCorrect(:,4), 20, 'Normalization','probability', 'FaceColor', 'g')
        title('Distribution of Prediction Confidences for Incorrectly vs Correctly Classified Examples');
        xlabel('Prediction Confidence'); ylabel('Frequency'); 

        % plot frequencies of incorrectly classified classes
        C = categorical(predictionsIncorrect(:,1),[0 1 2 3 4 5 6 7 8 9], {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'});
        figure;
        histogram(C, 'Normalization','probability')
        title('Frequency of Incorrectly Classified Classes');
        xlabel('Class'); ylabel('Frequency'); 
            clear C

        % plot confusion matrix
        testPredictedVectors = createMnistTargetVectors(predictions(:,1));
        testTargetVectors = createMnistTargetVectors(predictions(:,2));
        plotconfusion(testTargetVectors, testPredictedVectors)
        
    end
end

%% DISPLAY RANDOM SAMPLE OF MISCLASSIFIED DIGITS

if trainFinalNetwork == true
    if plotMetrics == true

        % construct matrix of incorrectly classified examples
        misclassfiedImages = testingData(:, misclassfiedIdx);
        yMisclass = predictions(misclassfiedIdx, 2);
        yhatMisclass = predictions(misclassfiedIdx, 1);
        yHatRaw = predictions(misclassfiedIdx, 4);
        
        % pick random sample of misclassified digits and display
        figure; displayMisclassified( misclassfiedImages, randi(size(misclassfiedImages, 2), 49, 1)', yMisclass, yhatMisclass, yHatRaw);
    
        % clear redundant variables from workspace
        clear yMisclass yhatMisclass
        
    end
end

%% LAB

% calculate number of examples in each class
% classesTest = [0:9; zeros(1,10)]';
% for c = 1:size(classesTest, 1)
%     for i = 1:size(testingLabels,1)        
%         if testingLabels(i) == classesTest(c, 1)
%             classesTest(c, 2) = classesTest(c, 2) + 1; 
%         end
%     end
% end 

    





    

