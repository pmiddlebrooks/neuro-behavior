%%
plot_color_grid(colors)

function plot_color_grid(rgbMatrix)
    % Function to plot a 4x4 grid of squares with given RGB colors
    % Input:
    %   rgbMatrix - A 16x3 matrix of RGB colors

    % Create a new figure
    figure;
    
    % Loop through each color and plot it
    for i = 1:size(rgbMatrix, 1)
        % Calculate subplot position
        subplot(4, 4, i);
        
        % Plot the color square
        rectangle('Position', [0 0 1 1], 'FaceColor', rgbMatrix(i,:), 'EdgeColor', 'none');
        
        % Adjust axis properties
        axis off;
        axis equal;
    end
end