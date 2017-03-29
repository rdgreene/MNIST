function [ classAccuracy, classError, MeanSquaredError, classMetrics, predictions  ] = evaluateMnistNN( inputData, inputLabels, hiddenLayerWeights, ouputLayerWeights, printOutput )
%Evaluates a 2 layer NN using data, labels, and weights inputs

    % convert input labels into target vectors
    inputLabels = createMnistTargetVectors(inputLabels);

    % init variable to track softmax error
    MeanSquaredError = 0;
    
    % get class predictions and class labels
    for y = 1:size(inputData, 2)

        % construct matrix off predicted and actual classes 
        yHat = netOutput( inputData(:,y), hiddenLayerWeights, ouputLayerWeights);
        [~, predictions(y, 1)] = max(yHat); % predicted class
        [~, predictions(y, 2)] = max(inputLabels(:,y)); % actual class
        predictions(y, 3) = predictions(y, 1) == predictions(y, 2); % prediction correct {1,0}
        predictions(y ,4) = max(yHat); % prediction confidence

        % add softmax error for example y
        MeanSquaredError = MeanSquaredError + norm(yHat-inputLabels(:,y),2);

    end
    
    % avaerage softmax error across all examples
    MeanSquaredError = MeanSquaredError / size(inputData, 2);

    % ...
    correct = sum(predictions(:,3));
    incorrect = sum(~predictions(:,3));
    classAccuracy = correct / (correct + incorrect);
    classError = 1 - classAccuracy;

    % calculate TP, FP, TN, FN for each class and store in matrix where rows
    % relate to class and columns relate to metric type {TP, FP, TN, FN}

    % init matrix for counting metrics
    classMetrics = zeros(10,4);

    for p = 1:size(predictions, 1)
        if predictions(p, 1) == predictions(p, 2) % if class was correctly predicted    
            for c = 1:10 % for each class...  
                if c == predictions(p, 1) % if c is  the predicted class    
                    classMetrics(c,1) = classMetrics(c,1)+1; % add to TP count for class         
                else              
                    classMetrics(c,3) = classMetrics(c,3)+1; % add to TN count for class    
                end
            end
        else  % if class was NOT correctly predicted  
            for c = 1:10 % for each class...    
                if c == predictions(p, 1) % if c is  the predicted class          
                    classMetrics(c,2) = classMetrics(c,2)+1; % add to FP count for class         
                elseif c == predictions(p,2) % if c is the actual class               
                    classMetrics(c,4) = classMetrics(c,4)+1; % add to FN count for class 
                else % otherwise...
                    classMetrics(c,3) = classMetrics(c,3)+1; % add to TN count for class
                end
            end
        end   
    end
    
    % convert class labels in 'predictions' back to original classes (i.e. change 10 to 0)
    for col = 1:2
        for p = 1:size(predictions,1) 
            if predictions(p, col) == 10;
                predictions(p, col) = 0;
            end
        end
    end
                   
    if printOutput == true
        fprintf('[Mean Squared Error: %.4f. ', MeanSquaredError)
        fprintf('Classification Error: %.4f]\n', classError)
    end
  
end

