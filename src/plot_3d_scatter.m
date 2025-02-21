
%% Plot on second monitor, half-width
xDim = 1;
yDim = 2;
zDim = 3;

fig = figure(figH);
% fig.Visible = 'off';
set(fig, 'Position', plotPos); clf; hold on;
title(titleM, 'interpreter', 'none')
if iDim > 2
    % scatter3(projSelect(plotFrames, xDim), projSelect(plotFrames, yDim), projSelect(plotFrames, zDim), 60, colorsForPlot, 'LineWidth', 2)
    scatter3(projSelect(plotFrames, xDim), projSelect(plotFrames, yDim), projSelect(plotFrames, zDim), 200, colorsForPlot, '.')
    % Variable to set the viewing angle
    azimuth = 30;  % Angle for rotation around the z-axis
    elevation = 50;  % Angle for elevation
    % Set the viewing angle
    view(azimuth, elevation);
    % Set axes ranges based on the data
    xlim([min(projSelect(plotFrames, xDim)), max(projSelect(plotFrames, xDim))]);
    ylim([min(projSelect(plotFrames, yDim)), max(projSelect(plotFrames, yDim))]);
    zlim([min(projSelect(plotFrames, zDim)), max(projSelect(plotFrames, zDim))]);
    % set(findall(fig,'-property','FontSize'),'FontSize',allFontSize) % adjust fontsize to your document
    % set(findall(fig,'-property','Box'),'Box','off') % optional
    xlabel(['D', num2str(xDim)]); ylabel(['D', num2str(yDim)]); zlabel(['D', num2str(zDim)])

elseif iDim == 2
    % scatter(projSelect(plotFrames, 1), projSelect(plotFrames, 2), 60, colorsForPlot, 'LineWidth', 2)
    scatter(projSelect(plotFrames, 1), projSelect(plotFrames, 2), 150, colorsForPlot, '.')
    xlabel(['D', num2str(1)]); ylabel(['D', num2str(2)]);
end

grid on;
% saveas(gcf, fullfile(paths.figurePath, [titleM, '.png']), 'png')
figure_pretty_things
% print('-dpdf', fullfile(paths.dropPath, [titleM, '.pdf']), '-bestfit')
print('-dpng', fullfile(paths.dropPath, [titleM, '.png']))
