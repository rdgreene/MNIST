function showMisclassifiedImages( images_data, images_to_display, y, yHat, yHatRaw)
%showImages: Displays images that have been flattened stored in a martix as columns
%
%       Inputs:
%           images_data: matrix of images flattened into column vectors
%           images_to_display: vector specifying column idices for image to be displayed
%           invert (optional): if 'invert' is input, images will be inverted
%
%       Important Note: Not compatible with non-square images (i.e. height != width)

% calculate dimesions of input images
d = sqrt(size(images_data, 1));

% calculate number of images being displayed
num_imgs = size(images_to_display, 2);
square_grid_size = ceil(sqrt(num_imgs));

% init counter
j = 1;

for image_number = images_to_display
    
    % reconstruct images
    for row = 0:d-1
        start = (d * row) + 1; % index of first pixel of current row of reconstructed image in inputted matrix column
        finish = d * (row + 1); % index of last pixel of current row of reconstructed image in inputted matrix column
        image(:, row+1) = images_data([start:finish], image_number); % construct current row for output 
    end

    %plot current image
    subplot(square_grid_size,square_grid_size,j)
    
    imshow(image)
    %title(sprintf('True Class: %d\nPredicted Class: %d\nClass Confidence: %.2f', [y(image_number) yHat(image_number) yHatRaw(image_number)]))
    title(sprintf('y: %d\nyHat: %d (%.2f)', [y(image_number) yHat(image_number) yHatRaw(image_number)]))
    
    % increment counter
    j=j+1;
    
end

