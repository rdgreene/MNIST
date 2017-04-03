function showConfusionMatrix (labelsOut, trainFeatures, tTrain)

fprintf('Show confusion matrix\n');
%% Show confusion matrix
isLabels = unique(tTrain);
[~,grpOOF] = ismember(labelsOut,isLabels); 
nLabels = numel(isLabels);
[n,p] = size(trainFeatures);
oofLabelMat = zeros(nLabels,n); 
idxLinear = sub2ind([nLabels n],grpOOF,(1:n)'); 

oofLabelMat(idxLinear) = 1; % Flags the row corresponding to the class 
[~,grpY] = ismember(tTrain,isLabels); 
YMat = zeros(nLabels,n); 
idxLinearY = sub2ind([nLabels n],grpY,(1:n)'); 
YMat(idxLinearY) = 1; 

figure;
plotconfusion(YMat,oofLabelMat);
h = gca;
h.XTickLabel = [num2cell(isLabels); {''}];
h.YTickLabel = [num2cell(isLabels); {''}];