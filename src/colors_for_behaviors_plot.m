%% Create a figure
figure;

% Plotting each color square and adding a label
for i = 1:17
    % Plot color square vertically
    patch([1, 2, 2, 1], [i, i, i+1, i+1], colors(i, :), 'EdgeColor', 'none');
    hold on;
    
    % Add label to the left of the square
    text(0.8, i + 0.5, behaviors{i}, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontSize', 8, 'interpreter', 'none');
end

axis([0 3 1 18]); % Adjust axis limits to fit the squares and labels vertically
axis off; % Hide axis for cleaner visualization
