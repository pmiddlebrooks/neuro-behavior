%% Create a figure
% figure;
% 
% % Plotting each color square and adding a label
% for i = 1:size(colors, 1)
%     % Plot color square vertically
%     patch([1, 2, 2, 1], [i, i, i+1, i+1], colors(i, :), 'EdgeColor', 'none');
%     hold on;
% 
%     % Add label to the left of the square
%     text(0.8, i + 0.5, behaviors{i}, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontSize', 8, 'interpreter', 'none');
% end
% 
% axis([0 3 1 18]); % Adjust axis limits to fit the squares and labels vertically
% axis off; % Hide axis for cleaner visualization
% 


% Number of behaviors
numBehaviors = length(behaviors);

% Figure settings
figure;
hold on;
axis off; % Hide axis

% Square settings
squareSize = 0.1; % Size of the color squares
spacing = 0.15;   % Vertical spacing to prevent touching

% Plot behavior names with color squares
for i = 1:numBehaviors
    yPos = -i * spacing; % Adjust spacing so squares don't touch
    
    % Draw colored square
    rectangle('Position', [-0.2, yPos, squareSize, squareSize], ...
              'FaceColor', colors(i, :), 'EdgeColor', 'none'); 
    
    % Display behavior name next to the square
    text(0, yPos + squareSize / 3, behaviors{i}, 'FontSize', 12, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'interpreter', 'none');
end

title('Behavior Colors Legend');
hold off;

%%
% % Define behavior names
% behaviors = {'Walking', 'Running', 'Jumping', 'Sitting', 'Eating'};
% 
% % Define corresponding colors (each row is an RGB value)
% colors = [ 
%     1, 0, 0;   % Red
%     0, 1, 0;   % Green
%     0, 0, 1;   % Blue
%     1, 1, 0;   % Yellow
%     0.5, 0, 0.5 % Purple
% ];

% Number of behaviors
numBehaviors = length(behaviors);

% Figure settings
figure(801); clf
hold on;
axis off; % Hide axes for clean display

% ###### Adjustable Parameters ######
squareSize = 0.25;    % Size of the color squares (Increase for bigger squares)
spacing = 0.6;        % Vertical spacing between rows (Increase to prevent overlap)
textSize = 14;        % Font size for labels
xOffset = 0.2;       % Space between square and label (Increase if text is too close)
textAdjust = 0.00;    % Fine-tune vertical alignment of text relative to squares
xStart = -0.15;        % Adjust to shift squares closer to left edge
% ###################################

% Set limits for proper spacing
xlim([-0.5, 1]); 
ylim([-numBehaviors * spacing - 0.5, 0.5]);

% Ensure correct aspect ratio (keeps squares as squares)
set(gca, 'DataAspectRatio', [1 1 1]);

% Plot behavior names with larger color squares and better spacing
for i = 1:numBehaviors
    yPos = -i * spacing; % Adjust vertical position
    
    % Define square coordinates
    xSquare = [xStart, xStart+squareSize, xStart+squareSize, xStart];
    ySquare = [yPos, yPos, yPos+squareSize, yPos+squareSize];
    
    % Draw square with fill()
    fill(xSquare, ySquare, colors(i, :), 'EdgeColor', 'none', 'LineWidth', 1.5);
    
    % % Display behavior name with adjusted spacing
    % text(-0.2 + squareSize + xOffset, yPos + squareSize / 3, behaviors{i}, ...
    %      'FontSize', textSize, 'FontWeight', 'bold', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'interpreter', 'none');
    % Adjust text y-position manually using textAdjust
    text(-0.2 + squareSize + xOffset, yPos + squareSize / 2 - textAdjust, behaviors{i}, ...
         'FontSize', textSize, 'FontWeight', 'bold', 'HorizontalAlignment', 'left', 'interpreter', 'none');
end

% Set title
% title('Behavior Colors Legend', 'FontSize', textSize+2, 'FontWeight', 'bold');

hold off;
