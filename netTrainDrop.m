function [ weightsHid, weightsOut, epochClassErrors, epochVectorErrors, epochValClassErrors, epochValVectorErrors ] = trainNN( hiddenLayerSize, batchSize, epochs, alpha, mu, dropout, trainingData, targetVectors, rngModifier, foldValData, foldValLabels)
% Use input data and parameters to train a neural network
	
    %{
 
    INPUTS: 
    hiddenLayerSize - number of neurons in hidden layer
    batchSize - size of batch to use for each backwards pass on training data
    epochs - number of epochs
    mu - momentum
    dropout - dropout coefficent, determines probability of a neuron in the hidden layer being 'dropped out'
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
    sd = inputs^-1/2;
    sdAdjust = sqrt(sd);
    rng(1); weightsHid = sdAdjust.*rand(hiddenLayerSize, inputs + 1); % weights connecting input layer to hidden layer
    rng(1); weightsOut = sdAdjust.*rand(outputs, hiddenLayerSize + 1); % weights sonnecting hidden layer to output layer

    % calculate number of batches to process
    batches = examples / batchSize;
    
    % init vector to track epoch error
    epochVectorErrors = [];
    
    fprintf('\nTraining Network with %d inputs, %d neurons in hidden layer, and %d outputs over %d epoch(s). [Learning rate = %.4f, Momentum = %.4f]', [inputs hiddenLayerSize outputs epochs alpha mu]);
    
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
        weightsOutUpdate = 0;
        weightsHidUpdate = 0;
        
        for b = 1:batches

        % WIP create boolean mask for dropout
        %m = 0.5; % dropout probability (set to zero to 'remove' dropout) REMOVE AFTER TESTING
        rng(e); dropMask = binornd(1,(1-dropout), hiddenLayerSize , 1);    
            
        % init vector for recording batch error 
        batchError = zeros(outputs,1);

            for x = batchSize*(b-1)+1:batchSize*b

            % Forward propagate the training vector through the network.
            trainingVector = trainingData(:, deck(x));
            zHidden = weightsHid*[1; trainingVector]; % 1 included for bias input
            aHidden = logisticSigmoid(zHidden);
            zOutput = weightsOut*[1; aHidden.*dropMask]; % 1 included for bias input, dropMask is optional and can be removed (drops out neurons in hidden layer)
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
        
        % ALTERNATIVE dropout 
        hiddenDelta = [dLogisticSigmoid(zHidden).*dropMask].*(weightsOut(:, 2:end)'*outputDelta);

        % back-propogate batch error step 2: calculate weight updates
        %WeightsOutUpdate = alpha.*outputDelta*[1 aHidden']; % 1 included for bias input
        %WeightsHidUpdate = alpha.*hiddenDelta*[1 trainingVector']; % 1 included for bias input
        
        % ALTERNATIVE WITH MOMENTUM back-propogate batch error step 2: calculate weight updates
        weightsOutUpdate = (alpha.*outputDelta*[1 aHidden'])+(mu.*weightsOutUpdate); % 1 included for bias input
        weightsHidUpdate = (alpha.*hiddenDelta*[1 trainingVector'])+(mu.*weightsHidUpdate); % 1 included for bias input

        % back-propogate batch error step 3: update weights
        weightsOut = weightsOut - weightsOutUpdate;
        weightsHid = weightsHid - weightsHidUpdate;

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
        epochVectorErrors(e) = epochError;
        
        
        % CALCULATE CLASSIFICATION ERROR FOR EPOCH
        
        countCorrect = 0;

        for x = 1:examples

            % predict classes using trained network weights and record precited
            % classes with actual class labels in a 2d array 
            yHat = netOutputDrop( trainingData(:,x), weightsHid, weightsOut, dropout);

            [~, yHatClass] = max(yHat);
            [~, yActualClass] = max(targetVectors(:,x));

            if yHatClass == yActualClass
                countCorrect = countCorrect+1;
            end
        end

        epochAccuracy = countCorrect/examples;
        epochClassError = 1-epochAccuracy;
        %fprintf('Classification error: %.4f. ', epochClassError);
        epochClassErrors(e) = epochClassError;
        
        if validate == 1
            % validate data here
            [~, valClassError, valVectorError, ~, ~] = netEvaluate( foldValData, foldValLabels, weightsHid, weightsOut, false );
            %fprintf('Epoch %d validated. %d examples processed. Vector error: %.4f. Classification error: %.4f\n', [e size(foldValLabels,1) valVectorError valClassError] )%[e size(foldValLabels) valClassError valVectorError])
        
            epochValClassErrors(e) = valClassError;
            epochValVectorErrors(e) = valVectorError;
            
        end
        
        if validate == 1
            fprintf('trained with %d examples & validated with %d examples. ', [examples size(foldValLabels,1)]);
            fprintf('TRAINING: [%.4f (ED), %.4f (C)]. VALIDATION: [%.4f (ED), %.4f (C)].', [epochError epochClassError valVectorError valClassError]);
        else
            fprintf('trained with %d examples. ', examples);
            fprintf('TRAINING: [%.4f (ED), %.4f (C)]. ', [epochError epochClassError])
        end
        
        % stop timing epoch
        epEnd = toc;
        
        fprintf('TIME: [%.2fs]', epEnd)
        
    end
    
    fprintf('\n');

end

