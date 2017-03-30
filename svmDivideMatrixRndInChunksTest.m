clc; clear;
%% unit test for funcion

matrixToTest = rand(6000);
labelsToTest = rand(6000,1);
[testChunksTrain, testChunksLabels] = svmDivideMatrixRndInChunks(matrixToTest,labelsToTest, 6);


for i = 1:size(testChunksTrain,2)
    
    toCompare = testChunksTrain{i};
    
    for ii = 1:size(testChunksTrain,2)
            if ii ~= i
                compareAgainst = testChunksTrain{ii};
                assert(isMatrixEquals(toCompare, compareAgainst) == false);                
            end
    end
end

%% assert that each chunk is different
for i = 1:size(testChunksLabels,2)
    
    toCompare = testChunksLabels{i};
    
    for ii = 1:size(testChunksLabels,2)
            if ii ~= i
                compareAgainst = testChunksLabels{ii};
                assert(isMatrixEquals(toCompare, compareAgainst) == false);                
            end
    end
end