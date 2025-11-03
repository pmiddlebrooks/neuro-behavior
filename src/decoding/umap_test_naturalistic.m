%%                     Test UMAP parameters on naturalistic data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script tests different UMAP parameters (min_dist, spread, n_neighbors)
% to fit naturalistic data and visualize the results

%% Specify main parameters
paths = get_paths;

% Data parameters
frameSize = .05;
slidingWindowSec = .2; % Duration of sliding window for spike summation

% Load naturalistic data
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 45 * 60; % seconds
opts.frameSize = frameSize;

% Get kinematics data for reference
getDataType = 'kinematics';
get_standard_data

% Get neural data
getDataType = 'spikes';
get_standard_data

% Curate behavior labels
[dataBhv, bhvID] = curate_behavior_labels(dataBhv, opts);

% Select brain area to analyze
areas = {'M23', 'M56', 'DS', 'VS'};
idM23 = find(strcmp(areaLabels, 'M23'));
idM56 = find(strcmp(areaLabels, 'M56'));
idDS = find(strcmp(areaLabels, 'DS'));
idVS = find(strcmp(areaLabels, 'VS'));


%% Apply sliding window to sum spikes
fprintf('Loaded naturalistic data: %d neurons, %d time points\n', size(dataMat, 2), size(dataMat, 1));
fprintf('Frame size: %.3f seconds\n', frameSize);
fprintf('Sliding window: %.3f seconds\n', slidingWindowSec);

% Calculate how many original bins fit in the sliding window
binsPerWindow = floor(slidingWindowSec / frameSize);
if binsPerWindow < 1
    binsPerWindow = 1;
end

% Create sliding window bins: sum spikes across the window centered on each bin
fprintf('Applying sliding window: %d bins per window\n', binsPerWindow);

% Calculate half-window size for centering
halfWindow = floor(binsPerWindow / 2);

% Initialize variables
nTimeBins = size(dataMat, 1);
nNeurons = size(dataMat, 2);

% Find valid bins where the sliding window is complete (not at edges)
validStart = halfWindow + 1;  % First bin with complete window on left
validEnd = nTimeBins - halfWindow;  % Last bin with complete window on right

% Number of valid bins
nValidBins = validEnd - validStart + 1;

% Initialize sliding window data matrix
dataMatSliding = zeros(nValidBins, nNeurons);
centerTimes = zeros(nValidBins, 1);

% Apply sliding window to valid bins only
validIdx = 1;
for binIdx = validStart:validEnd
    % Define window bounds (symmetric window)
    winStart = binIdx - halfWindow;
    winEnd = binIdx + halfWindow;
    
    % Sum spikes within the window
    dataMatSliding(validIdx, :) = sum(dataMat(winStart:winEnd, :), 1);
    
    % Store the center time (corresponding to original bin time)
    centerTimes(validIdx) = binIdx * frameSize;
    
    validIdx = validIdx + 1;
end

% Index bhvID to match only the valid sliding window bins
% bhvID is indexed by time, so we need to extract the bins corresponding to centerTimes
bhvIDSliding = bhvID(validStart:validEnd);

fprintf('Sliding window applied: %d valid bins (removed %d edge bins), %d neurons\n', ...
    size(dataMatSliding, 1), size(dataMat, 1) - size(dataMatSliding, 1), size(dataMatSliding, 2));
fprintf('Behavior labels indexed for sliding window data\n');
fprintf('First centerTime: %.3f s, Last centerTime: %.3f s\n', centerTimes(1), centerTimes(end));

% Define behavior labels and colors
bhv2Plot = {'investigate_1', 'investigate_2', 'investigate_3', ...
    'rear', 'dive_scrunch', 'paw_groom', 'face_groom_1', 'face_groom_2', ...
    'head_groom', 'contra_body_groom', 'ipsi_body groom', 'contra_itch', ...
    'ipsi_itch_1', 'contra_orient', 'ipsi_orient', 'locomotion'};
colors = colors_for_behaviors(codes);
colors2Plot = colors(2:end,:);
codes2Plot = codes(2:end);
%%
selectFrom = 'M56';  % Change to test different areas
switch selectFrom
    case 'M23'
        idSelect = idM23;
    case 'M56'
        idSelect = idM56;
    case 'DS'
        idSelect = idDS;
    case 'VS'
        idSelect = idVS;
end

fprintf('Selected %d neurons for area %s\n', length(idSelect), selectFrom);

% UMAP parameters to test
nDim = 3;  % 3D for visualization

% Test different parameter combinations
min_dist_values = [0.05, 0.1, 0.2, 0.5];
min_dist_values = [0.2, 0.5];
spread_values = [1.0, 1.5];
n_neighbors_values = [15, 30, 60];
min_dist_values = [0.2];
spread_values = [1.2];
n_neighbors_values = [30];

% Or test a specific combination
% min_dist_values = [0.1];
% spread_values = [1.0];
% n_neighbors_values = [15];

fprintf('\nTesting UMAP parameters:\n');
fprintf('min_dist: %s\n', mat2str(min_dist_values));
fprintf('spread: %s\n', mat2str(spread_values));
fprintf('n_neighbors: %s\n', mat2str(n_neighbors_values));

% Change to UMAP directory
cd(fullfile(paths.homePath, '/toolboxes/umapFileExchange (4.4)/umap/'))

% Monitor setup for plotting
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :);
if size(monitorPositions, 1) > 1
    monitorTwo = monitorPositions(2, :);
else
    monitorTwo = monitorOne;
end

% Test parameter combinations
configIdx = 1;
for min_dist = min_dist_values
    for spread = spread_values
        for n_neighbors = n_neighbors_values
            
            fprintf('\n===========================================\n');
            fprintf('Configuration %d:\n', configIdx);
            fprintf('min_dist=%.2f, spread=%.1f, n_neighbors=%d\n', min_dist, spread, n_neighbors);
            fprintf('===========================================\n');
            
            % Run UMAP on sliding window data
            fprintf('Running UMAP on sliding window data...\n');
            tic;
            [umapProjections, ~, ~, ~] = run_umap(zscore(dataMatSliding(:, idSelect)), 'n_components', nDim, ...
                'randomize', true, 'verbose', 'none', 'min_dist', min_dist, ...
                'spread', spread, 'n_neighbors', n_neighbors, 'ask', false);
            elapsedTime = toc;
            fprintf('UMAP completed in %.2f seconds\n', elapsedTime);
            
            % Plot the results
            figure(100 + configIdx);
            clf; hold on;
            
            % Set figure position (vertically fill second monitor, half-width horizontal)
            figWidth = monitorTwo(3) * 0.5;
            figHeight = monitorTwo(4);
            figX = monitorTwo(1) + (monitorTwo(3) - figWidth) / 2;
            figY = monitorTwo(2);
            set(gcf, 'Position', [figX, figY, figWidth, figHeight]);
                       
                % Plot data colored by behavior (use sliding window indexed behavior labels)
            for bhvIdx = 1:length(codes2Plot)
                % Map behavior code (1-based) to behavior index
                bhvCode = bhvIdx - 1;  % Convert to 0-based code
                bhvIndices = find(bhvIDSliding == bhvCode);
                if ~isempty(bhvIndices)
                    scatter3(umapProjections(bhvIndices, 1), umapProjections(bhvIndices, 2), umapProjections(bhvIndices, 3), ...
                        40, colors(bhvIdx, :), 'MarkerEdgeColor', colors2Plot(bhvIdx, :), 'LineWidth', 2, 'DisplayName', behaviors{bhvIdx});
                end
            end
            
            % Customize plot
            view(45, 30);
            xlabel('D1'); ylabel('D2'); zlabel('D3');
            title(sprintf('UMAP %s - min_dist=%.2f, spread=%.1f, nn=%d', selectFrom, min_dist, spread, n_neighbors), 'Interpreter', 'none');
            legend('Location', 'best', 'Interpreter', 'none');
            grid on;
            
            configIdx = configIdx + 1;
        end
    end
end

fprintf('\n===========================================\n');
fprintf('UMAP parameter testing complete!\n');
fprintf('Generated %d different UMAP configurations\n', configIdx - 1);
fprintf('===========================================\n');

% Return to original directory
cd(fullfile(paths.homePath, 'neuro-behavior/src/decoding'));


