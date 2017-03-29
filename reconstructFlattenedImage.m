function [ img ] = reconstructFlattenedImage( imageVector, show )
%takes square image that has been flattened into a vector and returns
%reconstructed square image matrix

% calculate dimesions of input images
d = sqrt(size(imageVector, 1));
    
    % reconstruct images
    for row = 0:d-1
        start = (d * row) + 1; % index of first pixel of current row of reconstructed image in inputted matrix column
        finish = d * (row + 1); % index of last pixel of current row of reconstructed image in inputted matrix column
        img(:, row+1) = imageVector([start:finish]); % construct current row for output 
    end

    if show == true
        %show current image
        imshow(image)
    end

end

