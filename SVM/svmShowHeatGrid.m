function svmShowHeatGrid(matrix, label)
size(matrix)

x = matrix;


xlabels = { '0.05', '0.1', '0.3', '0.5', '0.7', '1.0','2.0' };  % labels for box constraint
ylabels = { '0.05', '0.1', '0.3', '0.5', '0.7', '0.9',...      % labels for kernel scale
            '0.05', '0.1', '0.3', '0.5', '0.7', '0.9',...
            '0.05', '0.1', '0.3', '0.5', '0.7', '0.9',...
            '0.05', '0.1', '0.3', '0.5', '0.7', '0.9',...
            '0.05', '0.1', '0.3', '0.5', '0.7', '0.9',...
            '0.05', '0.1', '0.3', '0.5', '0.7', '0.9' };        

grid    =   x;


figure('position', [0, 0, 300, 500]);   % size of image
colormap('pink');                       % set colormap
imagesc(grid);                          % draw image and scale colormap to values range
colorbar;                               % show color scale on th side

xticks([1 2 3 4 5 6 7])                     % describe x axis
xticklabels(xlabels)
xlabel({'','Box Constraints',''},'FontSize',14,'Color','k')

yticks(1:24)                            % describe y axis
yticklabels(ylabels)
ylabel({'\fontsize{12}Hyperparameters search','',...
     '\fontsize{14}  Kernel Scales',''},...
     'Color','k')
title({'',strcat('\fontsize{12}Box Constraint x Kernel Scale ',label),''})
end