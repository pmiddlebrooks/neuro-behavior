%%
opts = neuro_behavior_options;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectEnd = 1 * 60 * 60; % seconds
% opts.frameSize = 1/60;
opts.frameSize = 1/60;

opts.minFiringRate = .5;

% getDataType = 'kinematics';
% opts.bhvDataPath = strcat(paths.bhvDataPath, 'animal_',animal,'/');
% getDataType = 'kinematics';
% kinData = load_data(opts, getDataType);
% kinData = load_data(opts, getDataType);
% get_standard_data
getDataType = 'kinematics';
get_standard_data


%% Get list of each body part tracked
bodyParts = unique(kinHeader(3,:));
bodyParts(strcmp(bodyParts, 'bodyparts')) = [];

%%
kinematics_velocities_relative_angles


%% PCA on the kinematics

[coeff, score, ~, ~, explained] = pca(kinData);
figure(33); plot(cumsum(explained)); grid on;



%% Find matching sequences
opts.window = round(.5 / opts.frameSize);
opts.thresholdCorr = .9;
opts.matchFrame = 1000;
dim = 1;
[matchFrames, matchCorr] = find_matched_sequences(kinData(:,dim), opts);
%% Umap on the kinematics

% [projSelect, ~, ~, ~] = run_umap(velocitiesAngles, 'n_components', 8, 'randomize', false, 'verbose', 'none', ...
%                         'min_dist', min_dist, 'spread', spread, 'n_neighbors', n_neighbors);
[projSelect, ~, ~, ~] = run_umap(kinData(1:size(dataMat, 1),:), 'n_components', 8, 'randomize', true, 'verbose', 'none');


%% Plot a movie of the body tracked positions over time
% Set up variables for plotting

% frames = 1000:2000;
% figure(2342); clf; hold on;
% for i = 1:length(bodyParts)
% iTrack = find(strcmp(headerData(3,:), bodyParts{i}), 2);
% plot(kinData{frames, iTrack(1)}, kinData{frames, iTrack(2)}, 'lineWidth', 2);
% end
% legend(bodyParts)
xMin = 10e3;
yMin = 10e3;
xMax = 0;
yMax = 0;
for i = 1:length(bodyParts)
iTrack = find(strcmp(kinHeader(3,:), bodyParts{i}), 2); % The first two columns of body part are x and y, respectively
xMin = floor(min([xMin; kinData{:, iTrack(1)}]));
yMin = floor(min([yMin; kinData{:, iTrack(2)}]));
xMax = floor(max([xMax; kinData{:, iTrack(1)}]));
yMax = floor(max([yMax; kinData{:, iTrack(2)}]));
% plot(kinData{frames, iTrack(1)}, kinData{frames, iTrack(2)}, 'lineWidth', 2);
end

snoutCol = find(strcmp(kinHeader(3,:), 'snout'), 2);
buttCol = find(strcmp(kinHeader(3,:), 'tail-base'), 2);

timeStamp = kinData{:,1} / opts.fsKinematics;

%%
delete(fullfile(paths.figurePath,'slidingWindowMovie.mp4'))

% Parameters
windowLength = 100; % Define the length of the sliding window
frameRate = 60; % Frame rate for the movie playback

% Assume dataSet1, dataSet2, dataSet3, dataSet4, dataSet5 are your 5 two-column matrices
% dataSets = {dataSet1, dataSet2, dataSet3, dataSet4, dataSet5};
maxRows = size(kinData, 1)/1000; % Find the minimum number of rows among datasets
maxRows = 500;

% Create a figure for plotting
figure(811); clf
% hold on;

% Prepare video writer
v = VideoWriter(fullfile(paths.figurePath,'slidingWindowMovie'), 'MPEG-4');
v.FrameRate = frameRate;
open(v);

startI = 1;
% Loop over the rows for sliding window
for i = startI:startI+maxRows
    % Define the window range
    % windowStart = max(1, i - windowLength + 1);
    % windowEnd = i;

    % Clear the plot for the new frame
    clf;
    hold on;
    set(gca, 'YDir', 'reverse')
xlim([xMin xMax])
ylim([yMin yMax])
    % Loop over each dataset and plot the windowed segment
    plot([kinData{i, snoutCol(1)} kinData{i, buttCol(1)}], [kinData{i, snoutCol(2)} kinData{i, buttCol(2)}], 'k')
    for j = 1:length(bodyParts)
jTrack = find(strcmp(headerData(3,:), bodyParts{j}), 2);
        % Ensure we don't exceed the available rows in the dataset
        % if windowStart <= size(kinData, 1)
            % plot(kinData{windowStart:windowEnd, jTrack(1)}, kinData{windowStart:windowEnd, jTrack(2)}, 'LineWidth', 2);
            scatter(kinData{i, jTrack(1)}, kinData{i, jTrack(2)}, 50, 'filled');
        % end
    end

    % Set plot limits and labels
    % xlim([min(cellfun(@(x) min(x(:,1)), dataSets)), max(cellfun(@(x) max(x(:,1)), dataSets))]);
    % ylim([min(cellfun(@(x) min(x(:,2)), dataSets)), max(cellfun(@(x) max(x(:,2)), dataSets))]);
    xlabel('X');
    ylabel('Y');
    % title(['Sliding Window from Row ' num2str(windowStart) ' to ' num2str(windowEnd)]);
    title(sprintf('Time: %.3f', timeStamp(i)));
    legend(['spine', bodyParts], 'Location', 'best');

    % Capture the frame
    frame = getframe(gcf);
    writeVideo(v, frame);
end

% Close the video writer
close(v);

% Display completion message
disp('Movie creation complete: slidingWindowMovie.mp4');






































%%
figure(43); clf; hold on;
% plot(sqrt(kinData.left_forepaw_x(1:500).^2 +kinData.left_forepaw_y(1:500).^2))
plot((kinData.left_forepaw_x(1:500) - kinData.snout_x(1:500)),  kinData.left_forepaw_y(1:500) - kinData.snout_y(1:500))
plot((kinData.right_forepaw_x(1:500) - kinData.snout_x(1:500)),  kinData.right_forepaw_y(1:500) - kinData.snout_y(1:500))







%%
idx = sort([2:3:size(kinData, 2), 3:3:size(kinData, 2)]);
kinPosition = table2array(kinData(:,idx));
kinVelocity = [];
for i = 1 : 2 : size(kinPosition, 2)
% Calculate differences in position (dx and dy) between consecutive samples
dx = diff(kinPosition(:,i)); % Differences in x
dy = diff(kinPosition(:,i+1)); % Differences in y
vel = sqrt(dx.^2 + dy.^2);

kinVelocity = [kinVelocity, [0; vel]];
end
%% Do PCA on the position and velocity data
[coeff, score, ~, ~, explained] = pca(zscore([kinPosition, kinVelocity], 1));
%% Plot the first few PCA components
nDim = 7;
window = 5 * 60 * 60 + (1 : 360);
figure(52); clf; hold on;
for d = 1 : nDim
plot(score(window, d));
end








%%
nosePos = [kinData.snout_x, kinData.snout_y];
tailPos = [kinData.tail_base_x, kinData.tail_base_y];
leftPaw = [kinData.left_forepaw_x, kinData.left_forepaw_y];
rightPaw = [kinData.right_forepaw_x, kinData.right_forepaw_y];

[leftPawY, leftPawX] = calculateSignedDistances(nosePos, tailPos, leftPaw);
[rightPawY, rightPawX] = calculateSignedDistances(nosePos, tailPos, rightPaw);


plotWindow = 1 : 300;
figure(43); clf; hold on;

yline(0, '--k')
plot(opts.frameSize * (0:length(plotWindow)-1), rightPawX(plotWindow), 'r')
plot(opts.frameSize * (0:length(plotWindow)-1), leftPawX(plotWindow), 'b')


function [distFromNose, distFromMidline] = calculateSignedDistances(nosePos, tailPos, thirdPoint)
    % Calculate the vector from the nose to each third point
    noseToThirdVec = thirdPoint - nosePos;
    
    % Calculate the unit vector for the line from the nose to the tail
    lineVec = tailPos - nosePos;
    unitLineVec = lineVec ./ sqrt(sum(lineVec.^2, 2));
    
    % Rotate the line vector by 90 degrees to get the perpendicular direction
    perpUnitLineVec = [-unitLineVec(:,2), unitLineVec(:,1)];
    
    % Calculate the signed distance from the nose along the line
    % This is the projection of the noseToThirdVec onto the line vector
    distFromNose = sum(noseToThirdVec .* unitLineVec, 2);
    
    % Calculate the signed perpendicular distance
    % This is the projection of the noseToThirdVec onto the perpendicular vector
    distFromMidline = sum(noseToThirdVec .* perpUnitLineVec, 2);
end