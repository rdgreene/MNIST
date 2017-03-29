function [ yHat ] = calculateOutput( inputVector, weightsHid, weightsOut)
% Calculates network outputs given inputted weights and vector

    yHat = logisticSigmoid(weightsOut*[1; logisticSigmoid(weightsHid*[1; inputVector])]);
    
end

