%% SVM_UMAP_FIGURES - Generate 3D UMAP projection figures for reach and naturalistic data
%
% Projects full reach and naturalistic sessions into 3-D UMAP space and creates
% three plots per brain area:
%   1. First sampleMin minutes with all points black (open circles)
%   2. Full session with all points black (open circles)
%   3. Full session with behavior-specified colors
%
% All plots use the same view/rotation for consistent presentation.
%
% Variables:
%   sampleMin - Number of minutes to plot in first figure
%   areas - Brain areas to analyze
%   dataMat - Neural activity matrix [time x neurons]
%   bhvID - Behavior labels [time x 1]
%   umapProj - UMAP projection [time x 3]
%   colors - Color matrix for behaviors

clear; close all;

%% Parameters
sampleMin = 5; % Minutes of data to show in first plot
nDim = 3; % UMAP dimensions (3D for visualization)
frameSize = 0.1; % seconds
lineWidth = 0.5; % Line width for scatter plot markers

% Brain areas
areas = {'M23', 'M56', 'DS', 'VS'};
areasToTest = 1:4;

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
[dataMatReach, idLabelsReach, areaLabelsReach] = neural_matrix_mark_data(dataR, opts);

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
    colorsReach(end,:) = [.85 .8 .75];
catch
    colorsReach = lines(nBehaviorsReach);
    colorsReach(end,:) = [.85 .8 .75];
end

fprintf('Loaded reach data: %d neurons, %d time points\n', size(dataMatReach, 2), size(dataMatReach, 1));

%% ==================== NATURALISTIC DATA ====================
fprintf('\n=== Loading Naturalistic Data ===\n');

optsNat = neuro_behavior_options;
optsNat.minActTime = 0.16;
optsNat.collectStart = 0 * 60 * 60; % seconds
optsNat.collectEnd = 45 * 60; % seconds
optsNat.frameSize = frameSize;

% Get neural data using get_standard_data script
getDataType = 'spikes';
get_standard_data

% Store naturalistic data in separate variables
dataMatNat = dataMat;
clear dataMat; % Clear to avoid confusion

% Curate behavior labels
[dataBhvNat, bhvIDNat] = curate_behavior_labels(dataBhv, optsNat);

% Set up areas (variables created by get_standard_data)
idListNat = {idM23, idM56, idDS, idVS};

% Naturalistic behavior colors
codesNat = unique(dataBhvNat.ID);
codesNat = codesNat(codesNat ~= -1);
try
    colorsNat = colors_for_behaviors(codesNat);
catch
    colorsNat = lines(max(codesNat) + 2);
end

fprintf('Loaded naturalistic data: %d neurons, %d time points\n', size(dataMatNat, 2), size(dataMatNat, 1));

%% ==================== PROCESS EACH DATA TYPE AND AREA ====================

dataTypes = {'reach', 'naturalistic'};
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
    else
        dataMat = dataMatNat;
        bhvID = bhvIDNat;
        idList = idListNat;
        colors = colorsNat;
        colorsAdjust = 2;
        sessionName = 'ag25290_112321';
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
        
        % Prepare data for UMAP
        dataArea = zscore(dataMat(:, idSelect));
        
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
                    spread = 1.3;
                    n_neighbors = 40;
                case 3
                    min_dist = 0.5;
                    spread = 1.2;
                    n_neighbors = 40;
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
                    min_dist = 0.2;
                    spread = 1.2;
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
        
        % Ensure bhvID matches dataMat length
        if length(bhvID) ~= size(dataMat, 1)
            fprintf('Warning: bhvID length (%d) does not match dataMat length (%d). Truncating...\n', ...
                length(bhvID), size(dataMat, 1));
            minLen = min(length(bhvID), size(dataMat, 1));
            bhvID = bhvID(1:minLen);
            umapProj = umapProj(1:minLen, :);
        end
        
        % Calculate sample indices (first sampleMin minutes)
        sampleFrames = round(sampleMin * 60 / frameSize);
        sampleIdx = 1:min(sampleFrames, size(umapProj, 1));
        
        % Create three plots with consistent view
        fprintf('Creating plots...\n');
        
        % Store view settings (will be set from first plot)
        viewAz = [];
        viewEl = [];
        
        % Plot 1: First sampleMin minutes, black open circles
        fig1 = figure('Position', [100, 100, 800, 800]);
        scatter3(umapProj(sampleIdx, 1), umapProj(sampleIdx, 2), umapProj(sampleIdx, 3), ...
            20, 'k', 'o', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'k', 'LineWidth', lineWidth);
        xlabel('UMAP 1');
        ylabel('UMAP 2');
        zlabel('UMAP 3');
        title(sprintf('%s - %s: First %d min (black)', upper(dataType), areaName, sampleMin));
        grid on;
        axis equal;
        
        % Set initial view and store it
        view(45, 30); % Default 3D view
        [viewAz, viewEl] = view;
        
        % Allow user to adjust view interactively
        fprintf('\n=== Adjust the 3D view for %s %s ===\n', dataType, areaName);
        fprintf('Rotate the plot to your desired view, then press any key to continue...\n');
        pause;
        [viewAz, viewEl] = view;
        fprintf('View set: Azimuth=%.1f, Elevation=%.1f\n', viewAz, viewEl);
        
        % Apply same view to all plots
        view(viewAz, viewEl);
        
        % Save first plot
        filename1 = sprintf('%s_%s_%s_%dmin_black.pdf', sessionName, dataType, areaName, sampleMin);
        filepath1 = fullfile(saveDir, filename1);
        exportgraphics(fig1, filepath1, 'ContentType', 'vector');
        fprintf('Saved: %s\n', filename1);
        
        % Plot 2: Full session, black open circles
        fig2 = figure('Position', [100, 100, 800, 800]);
        scatter3(umapProj(:, 1), umapProj(:, 2), umapProj(:, 3), ...
            20, 'k', 'o', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'k', 'LineWidth', lineWidth);
        xlabel('UMAP 1');
        ylabel('UMAP 2');
        zlabel('UMAP 3');
        title(sprintf('%s - %s: Full Session (black)', upper(dataType), areaName));
        grid on;
        axis equal;
        view(viewAz, viewEl); % Use same view
        
        % Save second plot
        filename2 = sprintf('%s_%s_%s_full_black.pdf', sessionName, dataType, areaName);
        filepath2 = fullfile(saveDir, filename2);
        exportgraphics(fig2, filepath2, 'ContentType', 'vector');
        fprintf('Saved: %s\n', filename2);
        
        % Plot 3: Full session, behavior colors
        fig3 = figure('Position', [100, 100, 800, 800]);
        
        % Get unique behavior IDs
        uniqueBhv = unique(bhvID);
        uniqueBhv = uniqueBhv(uniqueBhv >= 0); % Remove invalid labels
        
        % Plot each behavior with its color
        hold on;
        for b = 1:length(uniqueBhv)
            bhvCode = uniqueBhv(b);
            bhvMask = (bhvID == bhvCode);
            
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
            
            scatter3(umapProj(bhvMask, 1), umapProj(bhvMask, 2), umapProj(bhvMask, 3), ...
                20, plotColor, 'o', 'MarkerFaceColor', plotColor, 'MarkerEdgeColor', plotColor, 'LineWidth', lineWidth);
        end
        hold off;
        
        xlabel('UMAP 1');
        ylabel('UMAP 2');
        zlabel('UMAP 3');
        title(sprintf('%s - %s: Full Session (colored by behavior)', upper(dataType), areaName));
        grid on;
        axis equal;
        view(viewAz, viewEl); % Use same view
        
        % Save third plot
        filename3 = sprintf('%s_%s_%s_full_colored.pdf', sessionName, dataType, areaName);
        filepath3 = fullfile(saveDir, filename3);
        exportgraphics(fig3, filepath3, 'ContentType', 'vector');
        fprintf('Saved: %s\n', filename3);
        
        % Close figures to save memory
        close(fig1);
        close(fig2);
        close(fig3);
        
        fprintf('Completed %s: %s\n', dataType, areaName);
    end
end

fprintf('\n\n=== All figures generated and saved to: ===\n');
fprintf('%s\n', saveDir);
fprintf('\nComplete!\n');

