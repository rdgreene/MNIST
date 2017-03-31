xlabels = { '0.001', '0.005', '0.01', '0.05', '0.1' };  % labels for beta
ylabels = { '0.001', '0.005', '0.01', '0.05',...        % labels for gamma
            '0.001', '0.005', '0.01', '0.05',...
            '0.001', '0.005', '0.01', '0.05',...
            '0.001', '0.005', '0.01', '0.05',...
            '0.001', '0.005', '0.01', '0.05',...
            '0.001', '0.005', '0.01', '0.05' };        

grid    =   [];


figure('position', [0, 0, 300, 500]);   % size of image
colormap('pink');                       % set colormap
imagesc(grid);                          % draw image and scale colormap to values range
colorbar;                               % show color scale on th side

xticks([1 2 3 4 5])                     % describe x axis
xticklabels(xlabels)
xlabel({'','Beta',''},'FontSize',14,'Color','k')

yticks(1:24)                            % describe y axis
yticklabels(ylabels)
ylabel({'\fontsize{14}Gamma and Linear Research Routines','',...
     '\fontsize{10}  srchcha          srchhyb           srchgol          srchbre          srchcha          srchbac',''},...
     'Color','k')
title({'','\fontsize{12}Beta vs Gamma vs Search Routine',''})