%%                     Test UMAP parameters on reach data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script tests different UMAP parameters (min_dist, spread, n_neighbors)
% to fit reach data and visualize the results

%% Specify main parameters
paths = get_paths;

% Data parameters
reachCode = 2; % Behavior label for reaching animats
frameSize = .02;
slidingWindowSec = .1; % Duration of sliding window for spike summation

% Load reach data
reachDataFile = fullfile(paths.reachDataPath, 'Y4_06-Oct-2023 14_14_53_NeuroBeh.mat');
% reachDataFile = fullfile(paths.reachDataPath, 'reach_test.mat');
dataR = load(reachDataFile);

opts = neuro_behavior_options;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.collectEnd = round(min(dataR.R(end,1) + 5000, max(dataR.CSV(:,1)*1000)) / 1000);
opts.minFiringRate = .1;
opts.maxFiringRate = 70;
opts.frameSize = frameSize;

[dataMat, idLabels, areaLabels] = reach_neural_matrix(dataR, opts);

% Get behavior labels for reach task
bhvOpts = opts;
bhvID = define_reach_bhv_labels(reachDataFile, bhvOpts);

fprintf('Loaded reach data: %d neurons, %d time points\n', size(dataMat, 2), size(dataMat, 1));
fprintf('Frame size: %.3f seconds\n', frameSize);
fprintf('Sliding window: %.3f seconds\n', slidingWindowSec);

%% Apply sliding window to sum spikes
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
behaviors = {'pre-reach', 'reach', 'pre-reward', 'reward', 'post-reward', 'intertrial'};
nBehaviors = length(behaviors);
func = @sRGB_to_OKLab;
cOpts.exc = [0,0,0];
cOpts.Lmax = .8;
colors = maxdistcolor(nBehaviors,func, cOpts);
colors(end,:) = [.85 .8 .75];
colors(end,:) = [0 0 0];

% Select brain area to analyze
areas = {'M23', 'M56', 'DS', 'VS'};
idM23 = find(strcmp(areaLabels, 'M23'));
idM56 = find(strcmp(areaLabels, 'M56'));
idDS = find(strcmp(areaLabels, 'DS'));
idVS = find(strcmp(areaLabels, 'VS'));

%%
selectFrom = 'DS';  % Change to test different areas
switch selectFrom
    case 'M23'
        idSelect = idM23;
        min_dist_values = [0.3];
        spread_values = [1.2];
        n_neighbors_values = [30];
        min_dist_values = [0.4, 0.6 1];
        spread_values = [1 1.2 1.5 2.5];
        n_neighbors_values = [25 35 60];
    case 'M56'
        idSelect = idM56;
        min_dist_values = [.3];
        spread_values = [1.3];
        n_neighbors_values = [40]; 
        spread_values = [1.6];
        n_neighbors_values = [15]; 
    case 'DS'
        idSelect = idDS;
        min_dist_values = [0.4, 0.6 1];
        spread_values = [1 1.2 1.5 2.5];
        n_neighbors_values = [25 35 60];

        min_dist_values = [1];
        spread_values = [1.6];
        n_neighbors_values = [12];
    case 'VS'
        idSelect = idVS;
        min_dist_values = [0.1, 0.3, 0.5];
        spread_values = [1 1.2 1.5]  ;
        n_neighbors_values = [15 25 35];
        min_dist_values = [0.3];
        spread_values = [1.6];
        n_neighbors_values = [10];
end

fprintf('Selected %d neurons for area %s\n', length(idSelect), selectFrom);

% UMAP parameters to test
nDim = 3;  % 3D for visualization

% Test different parameter combinations

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
                'spread', spread, 'n_neighbors', n_neighbors);
            elapsedTime = toc;
            fprintf('UMAP completed in %.2f seconds\n', elapsedTime);

            % Subsample the inter-trial for plotting
                              counts = arrayfun(@(c) sum(bhvIDSliding==c), unique(bhvIDSliding));
                    maxSubsampleSize = round(mean(counts));  % Maximum samples per class when subsampling (categories with fewer samples use all their data, categories with more are subsampled to this max)

            %% Plot the results
            figure(100 + configIdx);
            clf; hold on;

            % Set figure position (vertically fill second monitor, half-width horizontal)
            figWidth = monitorTwo(3) * 0.5;
            figHeight = monitorTwo(4);
            figX = monitorTwo(1) + (monitorTwo(3) - figWidth) / 2;
            figY = monitorTwo(2);
            set(gcf, 'Position', [figX, figY, figWidth, figHeight]);

            % Plot data colored by behavior (use sliding window indexed behavior labels)
            for bhvIdx = 1:nBehaviors
                bhvIndices = find(bhvIDSliding == bhvIdx);
                if bhvIdx == 6
                    bhvIndices = bhvIndices(randperm(length(bhvIndices), maxSubsampleSize));
                end
                if ~isempty(bhvIndices)
                    if bhvIdx == 6
                        % Make intertrial (bhvIdx == 6) translucent edges only (no fill)
                        scatter3(umapProjections(bhvIndices, 1), umapProjections(bhvIndices, 2), umapProjections(bhvIndices, 3), ...
                            40, 'MarkerFaceColor', 'none', 'MarkerEdgeColor', colors(bhvIdx, :), 'LineWidth', 2, ...
                            'MarkerEdgeAlpha', 0.5, 'DisplayName', behaviors{bhvIdx});
                    else
                        scatter3(umapProjections(bhvIndices, 1), umapProjections(bhvIndices, 2), umapProjections(bhvIndices, 3), ...
                            40, colors(bhvIdx, :), 'MarkerEdgeColor', colors(bhvIdx, :), 'LineWidth', 2, 'DisplayName', behaviors{bhvIdx});
                    end
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

