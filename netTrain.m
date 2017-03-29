function [ weightsHid, weightsOut, epochTrainCE, epochTrainMSE, epochValCE, epochValMSE ] = trainNN( hiddenLayerSize, batchSize, epochs, alpha, mu, lambda, trainingData, targetVectors, rngModifier, foldValData, foldValLabels)
% Use input data and parameters to train a neural network
	
    %{
 
    INPUTS: 
    hiddenLayerSize - number of neurons in hidden layer
    batchSize - size of batch to use for each backwards pass on training data
    epochs - number of epochs
    mu - momentum 
    lambda - weight decay factor
    trainingData - matrix data for training neural net
    targetVectors - matrix of target vectors corresponding to each training examplefor, for calculating error
    rngModifier - rngModifier to ensure different 'shuffles' if using k-fold cross validation
    foldValData - matrix data for validating neural net [OPTIONAL]
    foldValLabels - vector of class labels corresponding to each testing example in foldValData [OPTIONAL]

    OUTPUTS:
    weightsHid - matrix of weights connecting inputs to hidden layer
    weightsOut - matrix of weights connecting hidden layer to output layer
    epochClassErrors - vector of training errors (classification) for each epoch
    epochVectorErrors  - vector of training errors (euclidian distance) for each epoch
    epochValClassErrors  - vector of validation errors (classification) for each epoch [OPTIONAL]
    epochValVectorErrors - vector of validation errors (euclidian distance) for each epoch [OPTIONAL]

    %}

    % check if validation data and labels have been supplied
    exist foldValData; A = ans == 1;
    exist foldValLabels; B = ans == 1;
    validate = A*B;
    
    % calculate #examples
    examples = size(trainingData, 2);
    
    % calculate #inputs
    inputs = size(trainingData, 1);
    
    % calculate #ouputs
    outputs = size(targetVectors, 1);
    
    % INIT NETWORK WEIGHTS
    
    % Init the weights for both layers (incl. bias weights).
    %rng(1); weightsHid = rand(hiddenLayerSize, inputs + 1); % weights connecting input layer to hidden layer
    %rng(1); weightsOut = rand(outputs, hiddenLayerSize + 1); % weights sonnecting hidden layer to output layer
    
    % scale weights based on size of layer
    %weightsHid = weightsHid./size(weightsHid, 2);
    %weightsOut = weightsOut./size(weightsOut, 2);
    
    % ALTERNATIVE INIT NETWORK WEIGHTS (Efficient Back Prop)
    sdHid = inputs^-1/2;
    sdAdjustHid = sqrt(sdHid);
    sdOut = hiddenLayerSize^-1/2;
    sdAdjustOut = sqrt(sdOut);
    rng(2); weightsHid = sdAdjustHid.*rand(hiddenLayerSize, inputs + 1); % weights connecting input layer to hidden layer
    rng(36); weightsOut = sdAdjustOut.*rand(outputs, hiddenLayerSize + 1); % weights sonnecting hidden layer to output layer

    % calculate number of batches to process
    batches = examples / batchSize;
    
    % init vector to track epoch error
    epochTrainMSE = [];
    
    fprintf('\nTraining Network with %d inputs, %d neurons in hidden layer, and %d outputs over %d epoch(s). [Learning rate = %.4f, Momentum = %.4f, Decay = %.4f]', [inputs hiddenLayerSize outputs epochs alpha mu lambda]);
    
    %fprintf('\nTraining Network with %d inputs, ', inputs);
    %fprintf('%d neurons in hidden layer, ', hiddenLayerSize);
    %fprintf('and %d outputs ', outputs);
    %fprintf('over %d epoch(s). ', epochs)
    %fprintf('[Learning rate = %.4f].\n', alpha);
    %fprintf('Training Network with %d neurons in hidden layer...\n\n', hiddenLayerSize);
    
    % RUN EPOCHS
    
    for e = 1:epochs

        fprintf('\nEpoch %d ', e);
        
        % start timing epoch
        tic;

        % shuffle 'deck' before forward propogation
        rng(e+rngModifier); deck = randperm(size(trainingData, 2));
        
        % init update variables (only required when using momentum)
        WeightsOutUpdate = 0;
        WeightsHidUpdate = 0;
        
        for b = 1:batches

        % init vector for recording batch error 
        batchError = zeros(outputs,1);

            for x = batchSize*(b-1)+1:batchSize*b

            % Forward propagate the training vector through the network.
            trainingVector = trainingData(:, deck(x));
            zHidden = weightsHid*[1; trainingVector]; % 1 included for bias input
            aHidden = logisticSigmoid(zHidden); 
            zOutput = weightsOut*[1; aHidden]; % 1 included for bias input
            aOutput = logisticSigmoid(zOutput);

            % calculate error
            yHat = aOutput;
            y = targetVectors(:, deck(x));
            error =  yHat - y;

            % update batch error vector
            batchError = batchError + error;

            end

        % calculate average error
        batchError = batchError./batchSize;

        % back-propogate batch error step 1: calculate deltas for output and
        % hidden layers
        outputDelta = dLogisticSigmoid(zOutput).*batchError;
        hiddenDelta = dLogisticSigmoid(zHidden).*(weightsOut(:, 2:end)'*outputDelta);

        % back-propogate batch error step 2: calculate weight updates
        %WeightsOutUpdate = alpha.*outputDelta*[1 aHidden']; % 1 included for bias input
        %WeightsHidUpdate = alpha.*hiddenDelta*[1 trainingVector']; % 1 included for bias input
        
        % ALTERNATIVE WITH MOMENTUM back-propogate batch error step 2: calculate weight updates
        WeightsOutUpdate = (alpha.*outputDelta*[1 aHidden'])+(mu.*WeightsOutUpdate); % 1 included for bias input
        WeightsHidUpdate = (alpha.*hiddenDelta*[1 trainingVector'])+(mu.*WeightsHidUpdate); % 1 included for bias input

        % back-propogate batch error step 3: update weights
        %weightsOut = weightsOut - WeightsOutUpdate;
        %weightsHid = weightsHid - WeightsHidUpdate;
        
        % ALTERNATIVE WITH WEIGHT DECAY back-propogate batch error step 3: update weights
        weightsOut = weightsOut - WeightsOutUpdate - (2*alpha*lambda*weightsOut);
        weightsHid = weightsHid - WeightsHidUpdate - (2*alpha*lambda*weightsHid);

        end

        % CALCULATE VECTOR ERROR FOR EPOCH
        
        epochError = 0;
        examples_tested = 0; % DIAG

        for x = 1:examples;

            trainingVector = trainingData(:, x);
            yHat = logisticSigmoid(weightsOut*[1; logisticSigmoid(weightsHid*[1; trainingVector])]);
            y = targetVectors(:, x);
            epochError = epochError + norm(yHat-y,2);

        end
        
        epochError = epochError/examples; %fprintf('Vector error: %.4f. ', epochError);

        % record vector error for each epoch
        epochTrainMSE(e) = epochError;
        
        
        % CALCULATE CLASSIFICATION ERROR FOR EPOCH
        
        countCorrect = 0;

        for x = 1:examples

            % predict classes using trained network weights and record precited
            % classes with actual class labels in a 2d array 
            yHat = netOutput( trainingData(:,x), weightsHid, weightsOut);

            [~, yHatClass] = max(yHat);
            [~, yActualClass] = max(targetVectors(:,x));

            if yHatClass == yActualClass
                countCorrect = countCorrect+1;
            end
        end

        epochAccuracy = countCorrect/examples;
        epochClassError = 1-epochAccuracy;
        %fprintf('Classification error: %.4f. ', epochClassError);
        epochTrainCE(e) = epochClassError;
        
        if validate == 1
            % validate data here
            [~, valCE, valMSE, ~, ~] = netEvaluate( foldValData, foldValLabels, weightsHid, weightsOut, false );
            %fprintf('Epoch %d validated. %d examples processed. Vector error: %.4f. Classification error: %.4f\n', [e size(foldValLabels,1) valVectorError valClassError] )%[e size(foldValLabels) valClassError valVectorError])
        
            epochValCE(e) = valCE;
            epochValMSE(e) = valMSE;
            
            fprintf('trained with %d examples & validated with %d examples. ', [examples size(foldValLabels,1)]);
            fprintf('TRAINING: [%.4f (MSE), %.4f (CE)]. VALIDATION: [%.4f (MSE), %.4f (CE)].', [epochError epochClassError valMSE valCE]);
        else
            fprintf('trained with %d examples. ', examples);
            fprintf('TRAINING: [%.4f (MSE), %.4f (CE)]. ', [epochError epochClassError])
        end
        
        % stop timing epoch
        epEnd = toc;
        
        fprintf('TIME: [%.2fs]', epEnd)
        
    end
    
    fprintf('\n');

end

