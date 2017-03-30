% nn coursework 1
clc; clear;

%% Load MNIST Training Data

xTrainImages = loadMNISTImages('train-images.idx3-ubyte');
tTrain = loadMNISTLabels('train-labels.idx1-ubyte');

%https://uk.mathworks.com/help/vision/examples/digit-classification-using-hog-features.html
hogTrainFeatures = svmHogFlattenedImages(xTrainImages);


%% Load MNIST Testing Data

xTestImages = loadMNISTImages('t10k-images.idx3-ubyte');
tTest = loadMNISTLabels('t10k-labels.idx1-ubyte');
%https://uk.mathworks.com/help/vision/examples/digit-classification-using-hog-features.html
hogTestFeatures = svmHogFlattenedImages(xTestImages);

% Transpose xTrainImages in line with the labels
xTrain = xTrainImages';

basicTemplate = templateSVM('Standardize',1,'ClassNames', {0,1,2,3,4,5,6,7,8,9});

linearTemplate = templateSVM('Standardize',1,'KernelFunction','linear',...
    'ClassNames',{0,1,2,3,4,5,6,7,8,9});

gaussianTemplate = templateSVM('Standardize',1,'KernelFunction',...
    'gaussian','ClassNames',{0,1,2,3,4,5,6,7,8,9});

polynomialTemplate = templateSVM('Standardize',1,'KernelFunction',...
    'polynomial','ClassNames',{0,1,2,3,4,5,6,7,8,9});


%% prepare templates for hyperparameter optimization


%% BoxConstraint
%A parameter that controls the maximum penalty imposed on margin-violating observations, and aids in preventing overfitting (regularization).
%If you increase the box constraint, then the SVM classifier assigns fewer support vectors. However, increasing the box constraint can lead to longer training times.

%% Kernel Scale
% The software divides all elements of the predictor matrix X by the value 
%of KernelScale. Then, the software applies the appropriate kernel norm to 
%compute the Gram matrix.

%% Standarize set to true
% The software centers and scales each column of the predictor data (X) by 
%the weighted column mean and standard deviation, respectively

%Kernel functions
kernelFunctions = {'linear';'gaussian';'polynomial'}
%Box constraint options
boxConstraints =  [1 2 4 5 7 10]
%Kernel scale options
kernelScales = [0.05 0.1 0.3 0.5 0.7 0.9]
% 'auto', then the software selects an appropriate scale factor using a heuristic procedure

%save all templates convination
svmHpTemplates = {};
index = 1;

for kernelFunction = 1:size(kernelFunctions, 1)
    for boxConstraint = 1:size(boxConstraints, 2)
        for kernelScale = 1:size(kernelScales, 2)
            fprintf('KernelFunction: s% BoxConstraint: %d KernelScale: %d \n' ...
                ,kernelFunctions{kernelFunction}  ...
                ,boxConstraints(boxConstraint) ...
                ,kernelScales(kernelScale));
            
                currentTemplate = templateSVM('Standardize',1,...
                    'KernelFunction',kernelFunctions{kernelFunction},...
                    'ClassNames',{0,1,2,3,4,5,6,7,8,9}, ...
                    'BoxConstraint',boxConstraints(boxConstraint),... 
                    'KernelScale',kernelScales(kernelScale));
                
                svmHpTemplates{index} = currentTemplate;
                index = index + 1;
            
        end
    end
end

save('svm_workspace')