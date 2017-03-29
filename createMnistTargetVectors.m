function [ targetVectors ] = createMnistTargetVectors( trainingLabels )
%Creates target vectors for MNIST dataset based on label data vector

    targetVectors = zeros(10, size(trainingLabels, 1));
    for i = 1: size(trainingLabels, 1)
        if trainingLabels(i) == 0
            targetVectors(10, i) = 1;
        else
            targetVectors(trainingLabels(i), i) = 1;
        end
    end;

end

