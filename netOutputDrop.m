function [ yHat ] = calculateOutput( inputVector, weightsHid, weightsOut, dropout)
% Calculates network outputs given inputted weights and vector

    yHat = logisticSigmoid((1-dropout)*weightsOut*[1; logisticSigmoid(weightsHid*[1; inputVector])]);
    
end

