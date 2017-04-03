function [matrixTrainChunks, labelTrainChunks] = svmDivideMatrixRndInChunks(trainMatrix, labelMatrix, noOfChunks) 
%% this function get matrix of training data and labels and divide it in
%  chunks, randomly shuffling the matrix
%  
%  function parameters: training matrix,label matrix and number of
%  chunks

%  returns: two list of matrices (chunks) for trining data and label matrix

    %get the size of the matrix in rows and cols
    [rows,cols]=size(trainMatrix);
    
    %random generation of index to pic rows
    indx = randperm(rows);
    
    %chop the rnd index into even required no of chunks
    rndChunks = reshape(indx, noOfChunks, rows/noOfChunks);
    
    %get row and cols of chunks
    [r, c] = size(rndChunks);
    
    % create cell list to collect chunks
    matrixTrainChunks = cell(1,r);
    labelTrainChunks = cell(1,1);
    
    %loop through each chunk of indexes
    for idx = 1:r
        %get rows chunk of indexes
        indexes = rndChunks(idx,:);
        %generate chunk matrix from rnd indexes
        matrixTrainChunks{idx} = trainMatrix(indexes,:);
        labelTrainChunks{idx} = labelMatrix(indexes,:);
    end 
end 