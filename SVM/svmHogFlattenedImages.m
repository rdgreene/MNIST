function trainingFeatures = hogFlattenedImages(images_data)
%https://uk.mathworks.com/help/vision/examples/digit-classification-using-hog-features.html
%showImages: Displays images that have been HOG flattened stored in a martix as columns


numImages = size(images_data, 2)

cellSize = [8 8];

% calculate dimesions of input images
d = sqrt(size(images_data, 1)); %keep
    
for i = 1:numImages

    % reconstruct images
    for row = 0:d-1
        start = (d * row) + 1; % index of first pixel of current row of reconstructed image in inputted matrix column
        finish = d * (row + 1); % index of last pixel of current row of reconstructed image in inputted matrix column
        image(:, row+1) = images_data([start:finish], i); % construct current row for output 
    end

    %imshow(image)
    
    %img = rgb2gray(image);

    % Apply pre-processing steps
    %image = imbinarize(image);

    [hog_8x8, vis8x8] = extractHOGFeatures(image, 'CellSize', cellSize);
    
    plot(vis8x8);
    if exist('trainingFeatures', 'var') == 0
        trainingFeatures = zeros(numImages, length(hog_8x8), 'single');
    end
    
    trainingFeatures(i, :) = hog_8x8;
    
end

