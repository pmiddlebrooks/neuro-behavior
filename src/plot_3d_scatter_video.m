%%

% Create a sample 3D matrix
plotRange = 30000:31000;
dataMatrix = projSelect(plotRange, 1:3);
colorMatrix = colorsForPlot(plotRange,:);

    % Number of interpolation points (smoothing)
    numSmoothPoints = 4;  % Add 10 interpolated points between original data points

% Call the function to create a scatter3 video with spline smoothing
createScatter3VideoWithTrail(dataMatrix, colorMatrix, numSmoothPoints, 'scatter3_video_spline.mp4');

function createScatter3VideoWithTrail(dataMatrix, colorMatrix, numSmoothPoints, outputVideoFile)
    % Function to create a video from 3D scatter plot data, with a trailing line
    % and smoothing using a spline interpolation.
    % Arguments:
    %   dataMatrix: a Nx3 matrix where each column represents x, y, z coordinates.
    %   outputVideoFile: the name of the output video file (e.g., 'scatter3_video_with_trail.mp4')

    % Validate input matrix size
    if size(dataMatrix, 2) ~= 3
        error('Input matrix must have exactly 3 columns (x, y, z coordinates).');
    end

    
    % Interpolate and smooth the data using spline interpolation
    [smoothedData, repeatedColors] = smoothDataAndExpandColors(dataMatrix, colorMatrix, numSmoothPoints);
    % expandedColors = repeatRows(colorsMatrix, numSmoothPoints+1);
    % Create video writer object
    videoWriter = VideoWriter(outputVideoFile, 'MPEG-4'); % Use 'MPEG-4' format
    open(videoWriter);

    % Create figure for plotting
    figureHandle = figure(777); clf
    set(figureHandle, 'Position', [100, 100, 1200, 800]);  % [x_pos, y_pos, width, height]
    hold all;
    
            % Variable to set the viewing angle
            azimuth = 60;  % Angle for rotation around the z-axis
            elevation = 20;  % Angle for elevation
            % Set the viewing angle
            view(azimuth, elevation);

            % Set up 3D scatter plot with moving marker
    scatterHandle = scatter3(smoothedData(1, 1), smoothedData(1, 2), smoothedData(1, 3), ...
        150, 'filled', 'MarkerFaceColor', repeatedColors(1,:), 'MarkerEdgeColor', 'k');
    scatterHandleTrail = scatter3(smoothedData(1, 1), smoothedData(1, 2), smoothedData(1, 3), ...
        90, repeatedColors(1,:), '.');
    xlabel('X-axis');
    ylabel('Y-axis');
    zlabel('Z-axis');
    title('3D Scatter Plot with Spline Smoothing and Trail');
    grid on;

    % Initialize plot for the trailing line
    % trailHandle = plot3(smoothedData(1, 1), smoothedData(1, 2), smoothedData(1, 3), '-', 'color', repeatedColors(1,:));
    
    % Define axis limits
    xlim([min(smoothedData(:, 1)), max(smoothedData(:, 1))]);
    ylim([min(smoothedData(:, 2)), max(smoothedData(:, 2))]);
    zlim([min(smoothedData(:, 3)), max(smoothedData(:, 3))]);

    % Loop through all smoothed data points and update the scatter plot and trail
    for i = 1:size(smoothedData, 1)
        % Update scatter plot with current data point (moving marker)
        set(scatterHandle, 'XData', smoothedData(i, 1), ...
                           'YData', smoothedData(i, 2), ...
                           'ZData', smoothedData(i, 3), ...
                           'MarkerFaceColor', repeatedColors(i,:), ...
                           'MarkerEdgeColor', 'k', ...
                           'SizeData', 150);
        set(scatterHandleTrail, 'XData', smoothedData(1:i, 1), ...
                           'YData', smoothedData(1:i, 2), ...
                           'ZData', smoothedData(1:i, 3), ...
                           'CData', repeatedColors(1:i,:), ...
                           'SizeData', 90);
        if i > 1
        % Update trail line with all points up to the current point
    plot3(smoothedData(i-1:i, 1), smoothedData(i-1:i, 2), smoothedData(i-1:i, 3),...
        '-', 'color', repeatedColors(i-1,:), 'LineWidth', 1);
        % set(trailHandle, 'XData', smoothedData(i-1:i, 1), ...
                         % 'YData', smoothedData(i-1:i, 2), ...
                         % 'ZData', smoothedData(i-1:i, 3), ...
                            % 'Color', repeatedColors(i-1,:));
        end
        % Capture the frame for the video
        frame = getframe(gcf);
        writeVideo(videoWriter, frame);
        
        pause(0.05); % Adjust speed of the plot
    end
    
    % Close the video writer object
    close(videoWriter);
    hold off;
    disp('Video creation with spline smoothing and trail completed.');
end

function [smoothedMatrix, repeatedColors] = smoothDataAndExpandColors(dataMatrix, colorMatrix, numSmoothPoints)
    % Function to smooth a 3-column dataMatrix using spline interpolation, and repeat
    % rows of colorMatrix to match the smoothed data points.
    % Arguments:
    %   dataMatrix: a Nx3 matrix where each column represents x, y, z coordinates.
    %   colorMatrix: a Nx3 matrix where each row represents color information (RGB or other).
    %   numSmoothPoints: the number of interpolated points to generate between each original point.
    % Output:
    %   smoothedMatrix: the interpolated and smoothed version of dataMatrix.
    %   repeatedColors: colorMatrix with rows repeated to match smoothedMatrix.
    
    % Validate input matrix size
    if size(dataMatrix, 2) ~= 3 || size(colorMatrix, 2) ~= 3
        error('Both dataMatrix and colorMatrix must have exactly 3 columns.');
    end
    
    % Ensure the dataMatrix and colorMatrix have the same number of rows
    if size(dataMatrix, 1) ~= size(colorMatrix, 1)
        error('dataMatrix and colorMatrix must have the same number of rows.');
    end

    % Number of original points
    numOriginalRows = size(dataMatrix, 1);
    
    % Generate the parameter t (progression) for the original data points
    t = 1:numOriginalRows;
    
    % Generate the new parameter t for the smoothed data points
    tFine = linspace(1, numOriginalRows, numOriginalRows + (numOriginalRows-1) * numSmoothPoints);
    
    % Perform spline interpolation for each dimension (x, y, z)
    smoothedX = spline(t, dataMatrix(:, 1), tFine);
    smoothedY = spline(t, dataMatrix(:, 2), tFine);
    smoothedZ = spline(t, dataMatrix(:, 3), tFine);
    
    % Combine the smoothed data into a single matrix
    smoothedMatrix = [smoothedX', smoothedY', smoothedZ'];
    
    % Number of rows in the smoothedMatrix
    numSmoothedRows = size(smoothedMatrix, 1);

    % Now repeat each row of colorMatrix to match the smoothedMatrix
    repeatedColors = [];
    
    % Loop through each original row and repeat it for the corresponding interpolated points
    for i = 1:numOriginalRows-1
        % Repeat the i-th row of colorMatrix for the points between dataMatrix(i) and dataMatrix(i+1)
        repeatedColors = [repeatedColors; repmat(colorMatrix(i, :), numSmoothPoints+1, 1)];
    end
    
    % Add the last row of colorMatrix (just once)
    repeatedColors = [repeatedColors; colorMatrix(end, :)];
    
    % Ensure that repeatedColors has the same number of rows as smoothedMatrix
    repeatedColors = repeatedColors(1:numSmoothedRows, :);
end

