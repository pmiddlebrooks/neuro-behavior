  
%% Plot on second monitor, half-width
        fig = figure(figH);
        set(fig, 'Position', plotPos); clf; hold on;
        title(titleM)
        if iDim > 2
            scatter3(projSelect(plotFrames, iDim-2), projSelect(plotFrames, iDim-1), projSelect(plotFrames, iDim), 60, colorsForPlot, 'LineWidth', 2)
            % Variable to set the viewing angle
            azimuth = 60;  % Angle for rotation around the z-axis
            elevation = 20;  % Angle for elevation
            % Set the viewing angle
            view(azimuth, elevation);
            % Set axes ranges based on the data
            xlim([min(projSelect(plotFrames, iDim-2)), max(projSelect(plotFrames, iDim-2))]);
            ylim([min(projSelect(plotFrames, iDim-1)), max(projSelect(plotFrames, iDim-1))]);
            zlim([min(projSelect(plotFrames, iDim)), max(projSelect(plotFrames, iDim))]);
            % set(findall(fig,'-property','FontSize'),'FontSize',allFontSize) % adjust fontsize to your document
            % set(findall(fig,'-property','Box'),'Box','off') % optional
         xlabel(['D', num2str(iDim-2)]); ylabel(['D', num2str(iDim-1)]); zlabel(['D', num2str(iDim)])

        elseif iDim == 2
            scatter(projSelect(plotFrames, 1), projSelect(plotFrames, 2), 60, colorsForPlot, 'LineWidth', 2)
         xlabel(['D', num2str(iDim-1)]); ylabel(['D', num2str(iDim)]);
       end
        grid on;
        % saveas(gcf, fullfile(paths.figurePath, [titleM, '.png']), 'png')
        figure_pretty_things
        print('-dpdf', fullfile(paths.figurePath, [titleM, '.pdf']), '-bestfit')
