gc%% SVM_UMAP_FIGURES - Generate 3D UMAP projection figures for reach and spontaneous data
%
% Projects full reach and spontaneous sessions into 3-D UMAP space using a
% sliding window approach and creates three plots per brain area:
%   1. First sampleMin minutes with all points black (open circles)
%   2. Full session with all points black (open circles)
%   3. Full session with behavior-specified colors
%
% All plots use the same view/rotation for consistent presentation.
%
% Uses sliding window to sum spikes across time windows (similar to
% umap_test_reach.m and umap_test_spontaneous.m).
%
% Variables:
%   sampleMin - Number of minutes to plot in first figure
%   frameSize - Original bin size (seconds)
%   slidingWindowSec - Duration of sliding window for spike summation (seconds)
%   areas - Brain areas to analyze
%   dataMat - Neural activity matrix [time x neurons] (original binned data)
%   dataMatSliding - Sliding window summed data [validBins x neurons]
%   bhvID - Behavior labels [time x 1] (original)
%   bhvIDSliding - Behavior labels indexed to sliding window data
%   umapProj - UMAP projection [validBins x 3]
%   colors - Color matrix for behaviors

%% Parameters
sampleMin = 3; % Minutes of data to show in first plot
nDim = 3; % UMAP dimensions (3D for visualization)
frameSize = 0.02; % seconds (original bin size)
slidingWindowSec = 0.1; % Duration of sliding window for spike summation
lineWidth = 1.5; % Line width for scatter plot markers
markerSize = 33;
subPropInter = 0.15; % Proportion of intertrial data to plot for reach data (bhvID = 6)

% Brain areas
areas = {'M23', 'M56', 'DS', 'VS'};
areasToTest = 3;

%% ==================== REACH DATA ====================



fprintf('\n=== Loading Reach Data ===\n');

sessionName = 'Y4_06-Oct-2023 14_14_53_NeuroBeh.mat';
reachDataFile = fullfile(paths.reachDataPath, sessionName);
dataR = load(reachDataFile);

opts = neuro_behavior_options;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.collectEnd = round(min(dataR.R(end,1) + 5000, max(dataR.CSV(:,1)*1000)) / 1000);
opts.minFiringRate = 0.1;
opts.maxFiringRate = 70;
opts.frameSize = frameSize;

% Get neural matrix
[dataMatReach, idLabelsReach, areaLabelsReach] = reach_neural_matrix(dataR, opts);

% Get behavior labels
bhvOpts = struct();
bhvOpts.frameSize = frameSize;
bhvOpts.collectStart = opts.collectStart;
bhvOpts.collectEnd = opts.collectEnd;
bhvIDReach = define_reach_bhv_labels(reachDataFile, bhvOpts);

% Set up areas
idM23Reach = find(strcmp(areaLabelsReach, 'M23'));
idM56Reach = find(strcmp(areaLabelsReach, 'M56'));
idDSReach = find(strcmp(areaLabelsReach, 'DS'));
idVSReach = find(strcmp(areaLabelsReach, 'VS'));
idListReach = {idM23Reach, idM56Reach, idDSReach, idVSReach};

% Reach behavior colors
behaviorsReach = {'pre-reach', 'reach', 'pre-reward', 'reward', 'post-reward', 'intertrial'};
nBehaviorsReach = length(behaviorsReach);
try
    func = @sRGB_to_OKLab;
    cOpts.exc = [0,0,0];
    cOpts.Lmax = .8;
    colorsReach = maxdistcolor(nBehaviorsReach, func, cOpts);
    colorsReach(end,:) = [0 0 0];
catch
    colorsReach = lines(nBehaviorsReach);
    colorsReach(end,:) = [0 0 0];
end

fprintf('Loaded reach data: %d neurons, %d time points\n', size(dataMatReach, 2), size(dataMatReach, 1));

%% ==================== SPONTANEOUS DATA ====================
fprintf('\n=== Loading Spontaneous Data ===\n');

optsNat = neuro_behavior_options;
optsNat.minActTime = 0.16;
optsNat.collectStart = 0 * 60 * 60; % seconds
optsNat.collectEnd = 45 * 60; % seconds
optsNat.frameSize = frameSize;

% Get neural data using get_standard_data script
getDataType = 'spikes';
get_standard_data

% Store spontaneous data in separate variables
dataMatNat = dataMat;
clear dataMat; % Clear to avoid confusion

% Curate behavior labels
% [dataBhvNat, bhvIDNat] = curate_behavior_labels(dataBhv, optsNat);
dataBhvNat = dataBhv;
bhvIDNat = bhvID;
% Set up areas (variables created by get_standard_data)
idListNat = {idM23, idM56, idDS, idVS};

% Spontaneous behavior colors
codesNat = unique(dataBhvNat.ID);
try
    colorsNat = colors_for_behaviors(codesNat);
catch
    colorsNat = lines(max(codesNat) + 2);
end

fprintf('Loaded spontaneous data: %d neurons, %d time points\n', size(dataMatNat, 2), size(dataMatNat, 1));


%% Setup paths
paths = get_paths;

% Add UMAP path
if exist(fullfile(paths.homePath, '/toolboxes/umapFileExchange (4.4)/umap/'), 'dir')
    umapPath = fullfile(paths.homePath, '/toolboxes/umapFileExchange (4.4)/umap/');
else
    error('UMAP toolbox not found. Please check path.');
end

% Create save directory
saveDir = fullfile(paths.dropPath, 'sfn2025', 'umap_figures');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end


%% ==================== PROCESS and plot EACH DATA TYPE AND AREA ====================
set(0, 'DefaultTextColor', '#D5A570');
set(0, 'DefaultAxesXColor', '#D5A570');
set(0, 'DefaultAxesYColor', '#D5A570');
set(0, 'DefaultAxesZColor', '#D5A570');
set(0, 'DefaultAxesFontSize', 30);
% set(0, 'DefaultAxesTickLength', [0.02 0.04]); % Default tick mark length
set(0, 'DefaultAxesLineWidth', 3); % Default tick mark linewidth = 4

% Set default grid color to match axes color (efdb7b)
% set(0, 'DefaultAxesGridColor', [239 219 123]/255);
% set(0, 'DefaultAxesXGrid', 'on');
% set(0, 'DefaultAxesYGrid', 'on');
% set(0, 'DefaultAxesZGrid', 'on');

dataTypes = {'reach', 'spontaneous'};
% dataTypes = {'reach'};
for dt = 1:length(dataTypes)
    dataType = dataTypes{dt};
    
    fprintf('\n\n========================================\n');
    fprintf('Processing %s data\n', upper(dataType));
    fprintf('========================================\n');
    
    % Select data based on type
     if strcmp(dataType, 'reach')
        dataMat = dataMatReach;
        bhvID = bhvIDReach;
        idList = idListReach;
        colors = colorsReach;
        colorsAdjust = 0;
        sessionName = 'Y4_06-Oct-2023_14_14_53';
        behaviors = behaviorsReach; % Behavior names for legend
    else
        dataMat = dataMatNat;
        bhvID = bhvIDNat;
        idList = idListNat;
        colors = colorsNat;
        colorsAdjust = 2;
        sessionName = 'ag25290_112321';
        % Get behavior names for spontaneous
        behaviors = cell(1, max(codesNat) + colorsAdjust);
        for i = 1:length(codesNat)
            code = codesNat(i);
            firstIdx = find(dataBhvNat.ID == code, 1);
            if ~isempty(firstIdx)
                behaviors{code + colorsAdjust} = dataBhvNat.Name{firstIdx};
            end
        end
    end
    
    % Process each brain area
    for areaIdx = areasToTest
        areaName = areas{areaIdx};
        fprintf('\n--- Processing %s: %s ---\n', dataType, areaName);
        
        % Select neurons for this area
        idSelect = idList{areaIdx};
        
        if length(idSelect) < nDim
            fprintf('Skipping %s: not enough neurons (%d < %d)\n', areaName, length(idSelect), nDim);
            continue;
        end
        
        fprintf('Selected %d neurons for area %s\n', length(idSelect), areaName);
        
        % Apply sliding window to sum spikes
        fprintf('Applying sliding window (%.3f s)...\n', slidingWindowSec);
        
        % Calculate how many original bins fit in the sliding window
        binsPerWindow = floor(slidingWindowSec / frameSize);
        if binsPerWindow < 1
            binsPerWindow = 1;
        end
        
        % Calculate half-window size for centering
        halfWindow = floor(binsPerWindow / 2);
        
        % Initialize variables
        nTimeBins = size(dataMat, 1);
        nNeurons = length(idSelect);
        
        % Find valid bins where the sliding window is complete (not at edges)
        validStart = halfWindow + 1;  % First bin with complete window on left
        validEnd = nTimeBins - halfWindow;  % Last bin with complete window on right
        
        % Number of valid bins
        nValidBins = validEnd - validStart + 1;
        
        % Initialize sliding window data matrix
        dataMatSliding = zeros(nValidBins, nNeurons);
        
        % Apply sliding window to valid bins only
        validIdx = 1;
        for binIdx = validStart:validEnd
            % Define window bounds (symmetric window)
            winStart = binIdx - halfWindow;
            winEnd = binIdx + halfWindow;
            
            % Sum spikes within the window
            dataMatSliding(validIdx, :) = sum(dataMat(winStart:winEnd, idSelect), 1);
            
            validIdx = validIdx + 1;
        end
        
        % Index bhvID to match only the valid sliding window bins
        bhvIDSliding = bhvID(validStart:validEnd);
        
        fprintf('Sliding window applied: %d valid bins (removed %d edge bins)\n', ...
            size(dataMatSliding, 1), size(dataMat, 1) - size(dataMatSliding, 1));
        
        % Prepare data for UMAP (z-score the sliding window data)
        dataArea = zscore(dataMatSliding);
        
        % Run UMAP in 3D
        fprintf('Running UMAP in 3D...\n');
        cd(umapPath);
        
        % UMAP parameters (from svm_decoding_compare.m)
        if strcmp(dataType, 'reach')
            switch areaIdx
                case 1
                    min_dist = 0.3;
                    spread = 1.2;
                    n_neighbors = 30;
                case 2
                    min_dist = 0.3;
                    spread = 1.6;
                    n_neighbors = 15;
                case 3
                    min_dist = 1;
                    spread = 1.6;
                    n_neighbors = 12;
                case 4
                    min_dist = 0.3;
                    spread = 1.2;
                    n_neighbors = 40;
            end
        else
            switch areaIdx
                case 1
                    min_dist = 0.1;
                    spread = 1;
                    n_neighbors = 15;
                case 2
                    min_dist = 0.2;
                    spread = 1.2;
                    n_neighbors = 30;
                case 3
                    min_dist = 0.3;
                    spread = 1.6;
                    n_neighbors = 30;
                case 4
                    min_dist = 0.2;
                    spread = 1.2;
                    n_neighbors = 30;
            end
        end
        
        warning('off', 'MATLAB:unrecognizedPragma');
        [umapProj, ~, ~, ~] = run_umap(dataArea, 'n_components', nDim, ...
            'randomize', true, 'verbose', 'none', 'min_dist', min_dist, ...
            'spread', spread, 'n_neighbors', n_neighbors, 'ask', false);
        warning('on', 'MATLAB:unrecognizedPragma');
        
        cd(fullfile(paths.homePath, 'neuro-behavior/src'));
        
        fprintf('UMAP projection complete: %d points in 3D\n', size(umapProj, 1));
        
        % Use sliding window indexed bhvID (already matched to umapProj length)
        % Calculate sample indices (first sampleMin minutes)
        % Note: sampleMin is based on original frameSize, but we're using sliding window data
        % So we need to convert: sampleMin minutes = sampleMin * 60 / slidingWindowSec sliding window bins
        sampleFrames = round(sampleMin * 60 / frameSize);
        startIdx = round(24*60/frameSize);
        sampleIdx = startIdx : startIdx + min(sampleFrames, size(umapProj, 1));
        
        % Create three plots with consistent view and axes
        fprintf('Creating plots...\n');
        
        % Calculate axis limits from all data to ensure consistency
        xLim = [min(umapProj(:, 1)), max(umapProj(:, 1))];
        yLim = [min(umapProj(:, 2)), max(umapProj(:, 2))];
        zLim = [min(umapProj(:, 3)), max(umapProj(:, 3))];
        
        % Store view settings (will be set from first plot - colorized)
        viewAz = [];
        viewEl = [];
        
        % Monitor setup for plotting
        monitorPositions = get(0, 'MonitorPositions');
        monitorOne = monitorPositions(1, :);
        if size(monitorPositions, 1) > 1
            monitorTwo = monitorPositions(2, :);
        else
            monitorTwo = monitorOne;
        end
        
        % Create single figure for this area/condition
        % Position on second monitor if available, with height = 90% of monitor height
        % Width = 1/3 of monitor width
        figWidth = round(monitorTwo(3) / 2);
        figHeight = round(monitorTwo(4) * 0.9);
        figX = monitorTwo(1) + (monitorTwo(3) - figWidth) / 2;
        figY = monitorTwo(2) + (monitorTwo(4) - figHeight) / 2;
        fig = figure('Position', [figX, figY, figWidth, figHeight]);
        
        %% Plot 1: Full session, behavior colors (FIRST - for setting view angle)
        clf(fig);
        axes;
        
        % Get unique behavior IDs (use sliding window indexed bhvID)
        % For spontaneous: project all data but only plot behaviors where bhvID != -1
        % For reach: project and plot all behaviors where bhvID >= 0
        uniqueBhv = unique(bhvIDSliding);
        if strcmp(dataType, 'spontaneous')
            uniqueBhv = uniqueBhv(uniqueBhv ~= -1); % Only plot behaviors, exclude -1
        else
            uniqueBhv = uniqueBhv(uniqueBhv >= 0); % Remove invalid labels for reach
        end
        
        % Plot each behavior with its color
        hold on;
        for b = 1:length(uniqueBhv)
            bhvCode = uniqueBhv(b);
            bhvMask = (bhvIDSliding == bhvCode);
            
            % For reach data, subsample intertrial (bhvID = 6) data
            if strcmp(dataType, 'reach') && bhvCode == 6
                bhvIndices = find(bhvMask);
                numToPlot = round(length(bhvIndices) * subPropInter);
                if numToPlot < length(bhvIndices)
                    rng(42); % Set seed for reproducibility
                    selectedIndices = bhvIndices(randperm(length(bhvIndices), numToPlot));
                    bhvMask = false(size(bhvMask));
                    bhvMask(selectedIndices) = true;
                end
            end
            
            % Get color for this behavior
            if strcmp(dataType, 'reach')
                colorIdx = bhvCode;
            else
                colorIdx = bhvCode + colorsAdjust;
            end
            
            if colorIdx > 0 && colorIdx <= size(colors, 1)
                plotColor = colors(colorIdx, :);
            else
                plotColor = [0.5, 0.5, 0.5]; % Gray for unmapped behaviors
            end
            
            % Get behavior name for legend
            if strcmp(dataType, 'reach')
                if bhvCode > 0 && bhvCode <= length(behaviors)
                    bhvName = behaviors{bhvCode};
                else
                    bhvName = sprintf('Behavior %d', bhvCode);
                end
            else
                if (bhvCode + 1) <= length(behaviors) && ~isempty(behaviors{bhvCode + 1})
                    bhvName = behaviors{bhvCode + colorsAdjust};
                else
                    bhvName = sprintf('Behavior %d', bhvCode);
                end
            end
            
            % For reach data, make intertrial (bhvID == 6) translucent
            if strcmp(dataType, 'reach') && bhvCode == 6
                % scatter3(umapProj(bhvMask, 1), umapProj(bhvMask, 2), umapProj(bhvMask, 3), ...
                %     markerSize, plotColor, 'o', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', plotColor, ...
                %     'LineWidth', lineWidth, 'MarkerEdgeAlpha', 0.5, 'DisplayName', bhvName);
                scatter3(umapProj(bhvMask, 1), umapProj(bhvMask, 2), umapProj(bhvMask, 3), ...
                    markerSize, plotColor, 'o', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', plotColor, ...
                    'LineWidth', lineWidth, 'DisplayName', bhvName);
            else
                scatter3(umapProj(bhvMask, 1), umapProj(bhvMask, 2), umapProj(bhvMask, 3), ...
                    markerSize, plotColor, 'o', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', plotColor, ...
                    'LineWidth', lineWidth, 'DisplayName', bhvName);
            end
        end
        hold off;
        
        xlabel('UMAP 1');
        ylabel('UMAP 2');
        zlabel('UMAP 3');
        title(sprintf('%s - %s: Full Session (colored by behavior)', upper(dataType), areaName));
        grid on;
        axis equal;
        xlim(xLim); ylim(yLim); zlim(zLim); % Set consistent axes
        % legend('Location', 'best', 'Interpreter', 'none'); % Add legend for colorized plot
        
        % Set initial view and store it
        view(45, 30); % Default 3D view
        [viewAz, viewEl] = view;
        
        % Allow user to adjust view interactively on colorized plot
        fprintf('\n=== Adjust the 3D view for %s %s (colorized plot) ===\n', dataType, areaName);
        fprintf('Rotate the plot to your desired view, then press any key to continue...\n');
        pause;
        [viewAz, viewEl] = view;
        fprintf('View set: Azimuth=%.1f, Elevation=%.1f\n', viewAz, viewEl);
        
        % Apply same view to this plot
        view(viewAz, viewEl);
        
        % Save first plot (colorized)
        filename1 = sprintf('%s_%s_%s_full_colored.eps', sessionName, dataType, areaName);
        filepath1 = fullfile(saveDir, filename1);
        print(fig, '-depsc', '-vector', filepath1);
        fprintf('Saved: %s\n', filename1);
        
        % Plot 2: Full session, black open circles
        clf(fig);
        axes;
        scatter3(umapProj(:, 1), umapProj(:, 2), umapProj(:, 3), ...
            markerSize, 'k', 'o', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'k', 'LineWidth', lineWidth, 'MarkerEdgeAlpha', 0.5);
        xlabel('UMAP 1');
        ylabel('UMAP 2');
        zlabel('UMAP 3');
        title(sprintf('%s - %s: Full Session (black)', upper(dataType), areaName));
        grid on;
        axis equal;
        xlim(xLim); ylim(yLim); zlim(zLim); % Set consistent axes
        view(viewAz, viewEl); % Use same view
        
        % Save second plot
        filename2 = sprintf('%s_%s_%s_full_black.eps', sessionName, dataType, areaName);
        filepath2 = fullfile(saveDir, filename2);
        print(fig, '-depsc', '-vector', filepath2);
        fprintf('Saved: %s\n', filename2);
        
        % Plot 3: First sampleMin minutes, black open circles
        clf(fig);
        axes;
        scatter3(umapProj(sampleIdx, 1), umapProj(sampleIdx, 2), umapProj(sampleIdx, 3), ...
            markerSize, 'k', 'o', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'k', 'LineWidth', lineWidth, 'MarkerEdgeAlpha', 0.5);
        xlabel('UMAP 1');
        ylabel('UMAP 2');
        zlabel('UMAP 3');
        title(sprintf('%s - %s: First %d min (black)', upper(dataType), areaName, sampleMin));
        grid on;
        axis equal;
        xlim(xLim); ylim(yLim); zlim(zLim); % Set consistent axes
        view(viewAz, viewEl); % Use same view
        
        % Save third plot
        filename3 = sprintf('%s_%s_%s_%dmin_black.eps', sessionName, dataType, areaName, sampleMin);
        filepath3 = fullfile(saveDir, filename3);
        print(fig, '-depsc', '-vector', filepath3);
        fprintf('Saved: %s\n', filename3);
        
        % Close figure to save memory
        close(fig);
        
        fprintf('Completed %s: %s\n', dataType, areaName);
    end
end

fprintf('\n\n=== All figures generated and saved to: ===\n');
fprintf('%s\n', saveDir);
fprintf('\nComplete!\n');

