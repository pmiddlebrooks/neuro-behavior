%%
% Criticality Sliding Window Analysis Script using PSID Latents (d2 only)
% Unified script for analyzing both reach data and naturalistic data
% Uses PSID to extract low-dimensional latents, projects back to neural space,
% then calculates d2 criticality measures using sliding window approach
%
% Variables:
%   psidInputType - 'kinematics' or 'behavior' for PSID input
%   nDimBhv - Number of behavior-related dimensions (also equals nDimNonBhv)
%   yPredBhv - Neural space projection from behavior-related latents
%   yPredNonBhv - Neural space projection from non-behavior latents
%   d2Bhv - d2 values for behavior-related latents (cell array: {area}{nDim})
%   d2NonBhv - d2 values for non-behavior latents (cell array: {area}{nDim})

paths = get_paths;

% =============================    Configuration    =============================
% Data type selection
dataType = 'naturalistic';  % 'reach' or 'naturalistic'

% Sliding window size (seconds)
slidingWindowSize = 20;

% Flags
loadExistingResults = false;
makePlots = true;

% Analysis flags
analyzeD2 = true;      % compute d2 (always true for this script)

% PSID configuration
psidInputType = 'kinematics';  % 'kinematics' or 'behavior' - what to use as PSID input z
nDimBhv = 3;           % Number of behavior-related dimensions (nDimNonBhv = nDimBhv)
nDimNonBhv = nDimBhv;  % Number of non-behavior dimensions (must equal nDimBhv)
psidI = 10;            % PSID parameter i (number of past lags)

% Analysis parameters
minSegmentLength = 50;
minSpikesPerBin = 3;
maxSpikesPerBin = 50;
minBinsPerWindow = 1000;

% Areas to analyze
areasToTest = 1:4;
areasToPlot = areasToTest;

% Optimal bin/window size search parameters
candidateFrameSizes = [.02 .03 .04 0.05, .075, 0.1 .15 .2];
candidateWindowSizes = [30, 45, 60, 90, 120];
windowSizes = repmat(slidingWindowSize, 1, 4);
pOrder = 10;
critType = 2;
d2StepSize = repmat(.2, 1, 4);

% =============================    Data Loading    =============================
fprintf('\n=== Loading %s data ===\n', dataType);

% Create filename suffix based on PSID input type
filenameSuffix = sprintf('_psid_%s', psidInputType);

    opts = neuro_behavior_options;
    opts.collectStart = 0;
    opts.frameSize = .001;
    opts.minFiringRate = .1;
    opts.maxFiringRate = 70;
    opts.firingRateCheckTime = 5 * 60;

if strcmp(dataType, 'reach')
    % Load reach data
    reachDataFile = fullfile(paths.reachDataPath, 'Y4_06-Oct-2023 14_14_53_NeuroBeh.mat');
    
    [~, dataBaseName, ~] = fileparts(reachDataFile);
    saveDir = fullfile(paths.dropPath, 'reach_task/results', dataBaseName);
    if ~exist(saveDir, 'dir'); mkdir(saveDir); end
    
    resultsPath = fullfile(saveDir, sprintf('criticality_sliding_window_ar%s_win%d.mat', filenameSuffix, slidingWindowSize));
    
    dataR = load(reachDataFile);
    
    opts.collectEnd = round(min(dataR.R(end,1) + 5000, max(dataR.CSV(:,1)*1000)) / 1000);
    
    [dataMat, idLabels, areaLabels] = neural_matrix_mark_data(dataR, opts);
    areas = {'M23', 'M56', 'DS', 'VS'};
    idM23 = find(strcmp(areaLabels, 'M23'));
    idM56 = find(strcmp(areaLabels, 'M56'));
    idDS = find(strcmp(areaLabels, 'DS'));
    idVS = find(strcmp(areaLabels, 'VS'));
    idMatIdx = {idM23, idM56, idDS, idVS};
    idLabel = {idLabels(idM23), idLabels(idM56), idLabels(idDS), idLabels(idVS)};
    
    % Get behavior labels for reach task
    bhvOpts = struct();
    bhvOpts.frameSize = opts.frameSize;
    bhvOpts.collectStart = opts.collectStart;
    bhvOpts.collectEnd = opts.collectEnd;
    bhvID = define_reach_bhv_labels(reachDataFile, bhvOpts);
    
    % Get kinematics data for PSID (if needed)
    if strcmp(psidInputType, 'kinematics')
        jsX = zscore(dataR.NIDATA(7,2:end));
        jsY = zscore(dataR.NIDATA(8,2:end));
        kinData = [jsX(:), jsY(:)];
        % Downsample kinData to match frameSize
        if opts.frameSize > .001
            binsPerFrame = round(opts.frameSize * 1000);
            numFrames = floor(size(kinData,1) / binsPerFrame);
            kinDataDown = zeros(numFrames, size(kinData,2));
            for i = 1:numFrames
                idxStart = (i-1)*binsPerFrame + 1;
                idxEnd = i*binsPerFrame;
                kinDataDown(i,:) = mean(kinData(idxStart:idxEnd,:), 1, 'omitnan');
            end
            kinData = kinDataDown(1:size(dataMat, 1),:);
        end
    end
    
elseif strcmp(dataType, 'naturalistic')
    % Load naturalistic data
    getDataType = 'spikes';
    opts.collectEnd = 45 * 60; % seconds
    get_standard_data
    
    areas = {'M23', 'M56', 'DS', 'VS'};
    idMatIdx = {idM23, idM56, idDS, idVS};
    idLabel = {idLabels(idM23), idLabels(idM56), idLabels(idDS), idLabels(idVS)};
    
    % Create save directory for naturalistic data
    saveDir = fullfile(paths.dropPath, 'criticality/results');
    if ~exist(saveDir, 'dir'); mkdir(saveDir); end
    resultsPath = fullfile(saveDir, sprintf('criticality_sliding_window_ar%s_win%d.mat', filenameSuffix, slidingWindowSize));
    
    % Get behavior labels for naturalistic data
    if strcmp(psidInputType, 'behavior')
        % Curate behavior labels
        [dataBhv, bhvID] = curate_behavior_labels(dataBhv, opts);
    end
    
    % Get kinematics data for PSID (if needed)
    if strcmp(psidInputType, 'kinematics')
        getDataType = 'kinematics';
        get_standard_data
        % kinData should now be available from get_standard_data
    end

else
    error('Invalid dataType. Must be ''reach'' or ''naturalistic''');
end
fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS))

%% =============================    Analysis    =============================
fprintf('\n=== %s Data Analysis with PSID ===\n', dataType);
fprintf('PSID input type: %s\n', psidInputType);
fprintf('Number of behavior dimensions: %d\n', nDimBhv);
fprintf('Number of non-behavior dimensions: %d\n', nDimNonBhv);

% Step 1: Find optimal parameters for each area
fprintf('\n--- Step 1: Finding optimal parameters ---\n');
optimalBinSize = zeros(1, length(areas));
optimalWindowSize = zeros(1, length(areas));

for a = areasToTest
    aID = idMatIdx{a};
    thisDataMat = dataMat(:, aID);
    [optimalBinSize(a), optimalWindowSize(a)] = ...
        find_optimal_bin_and_window(thisDataMat, candidateFrameSizes, candidateWindowSizes, minSpikesPerBin, maxSpikesPerBin, minBinsPerWindow);
    fprintf('Area %s: optimal bin size = %.3f s, optimal window size = %.1f s\n', areas{a}, optimalBinSize(a), optimalWindowSize(a));
end

warning('For now, using defined binsizes across areas instead of optimal for each area')
acrossAreaBinSize = .03;
optimalBinSize = repmat(acrossAreaBinSize, 1, 4);
% Use optimal bin sizes for each area
d2StepSizeData = d2StepSize;
d2WindowSizeData = windowSizes;
validMask = isfinite(optimalBinSize(areasToTest)) & (optimalBinSize(areasToTest) > 0);
areasToTest = areasToTest(validMask);

% Initialize results storage
% d2Bhv{area}{nDim} - d2 for behavior latents using 1:nDim dimensions
% d2NonBhv{area}{nDim} - d2 for non-behavior latents using 1:nDim dimensions
% d2BhvSingle{area}{dim} - d2 for individual behavior latent dimension
% d2NonBhvSingle{area}{dim} - d2 for individual non-behavior latent dimension
d2Bhv = cell(1, length(areas));
d2NonBhv = cell(1, length(areas));
d2BhvSingle = cell(1, length(areas));
d2NonBhvSingle = cell(1, length(areas));
startS = cell(1, length(areas));
popActivityBhv = cell(1, length(areas));
popActivityNonBhv = cell(1, length(areas));

for a = areasToTest
    fprintf('\nProcessing area %s (%s)...\n', areas{a}, dataType);
    tic;
    aID = idMatIdx{a};
    stepSamples = round(d2StepSizeData(a) / optimalBinSize(a));
    winSamples = round(d2WindowSizeData(a) / optimalBinSize(a));
    
    % Skip this area if there aren't enough samples
    if winSamples < minSegmentLength
        fprintf('\nSkipping: Not enough data in %s (%s)...\n', areas{a}, dataType);
        continue
    end
    
    % Bin data to optimal bin size
    aDataMat = neural_matrix_ms_to_frames(dataMat(:, aID), optimalBinSize(a));
    numTimePoints = size(aDataMat, 1);
    numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;
    
    % ========== Step 2: Run PSID to get latents ==========
    fprintf('  Running PSID...\n');
    
    % Prepare neural data
    y = zscore(aDataMat);
    
    % Prepare PSID input (z) based on psidInputType
    if strcmp(psidInputType, 'kinematics')
        % Use kinematics
        if strcmp(dataType, 'reach')
            % Downsample kinData to match aDataMat size
            if size(kinData, 1) ~= numTimePoints
                % Interpolate or downsample as needed
                if size(kinData, 1) > numTimePoints
                    % Downsample
                    downsampleFactor = size(kinData, 1) / numTimePoints;
                    kinDataArea = zeros(numTimePoints, size(kinData, 2));
                    for t = 1:numTimePoints
                        startIdx = round((t-1) * downsampleFactor) + 1;
                        endIdx = round(t * downsampleFactor);
                        endIdx = min(endIdx, size(kinData, 1));
                        kinDataArea(t, :) = mean(kinData(startIdx:endIdx, :), 1, 'omitnan');
                    end
                else
                    % Interpolate
                    oldTime = 1:size(kinData, 1);
                    newTime = linspace(1, size(kinData, 1), numTimePoints);
                    kinDataArea = interp1(oldTime, kinData, newTime, 'linear', 'extrap');
                end
            else
                kinDataArea = kinData;
            end
        else
            % Naturalistic - kinData should already match
            kinDataArea = kinData(1:numTimePoints, :);
        end
        z = zscore(kinDataArea);
    else
        % Use behavior labels (one-hot encoded)
        uniqueBhv = unique(bhvID);
        uniqueBhv = uniqueBhv(uniqueBhv >= 0);  % Remove invalid labels
        nBehaviors = length(uniqueBhv);
        
        % Create one-hot encoding
        bhvOneHot = zeros(numTimePoints, nBehaviors);
        bhvIDArea = bhvID(1:numTimePoints);  % Match length
        for b = 1:nBehaviors
            bhvOneHot(:, b) = (bhvIDArea == uniqueBhv(b));
        end
        z = zscore(bhvOneHot);
    end
    
    % PSID parameters
    nx = nDimBhv * 2;  % Total latent dimensions
    n1 = nDimBhv;      % Behavior-related dimensions
    i = psidI;
    
    % Run PSID
    idSys = PSID(y, z, nx, n1, i);
    
    % Get latents and neural projections
    [zPred, yPred, xPred] = PSIDPredict(idSys, y);
    
    % xPred(:, 1:nDimBhv) are behavior-related latents
    % xPred(:, nDimBhv+1:end) are non-behavior latents
    % yPred is the full neural space projection
    
    fprintf('  PSID complete: %d behavior latents, %d non-behavior latents\n', nDimBhv, size(xPred, 2) - nDimBhv);
    
    % ========== Step 3: Learn mapping from latents to neural space ==========
    % Learn linear mapping: yPred = xPred * W (where W maps latents to neural space)
    % This allows us to project subsets of latents back to neural space
    fprintf('  Learning latent-to-neural mapping...\n');
    
    % Learn full mapping from all latents to neural space
    % yPred = xPred * W, solve for W using least squares
    W_full = (xPred' * xPred) \ (xPred' * yPred);
    
    % ========== Step 4: Iteratively calculate d2 for different numbers of dimensions ==========
    fprintf('  Calculating d2 for different numbers of dimensions...\n');
    
    % Initialize storage for this area
    d2Bhv{a} = cell(1, nDimBhv);
    d2NonBhv{a} = cell(1, nDimNonBhv);
    d2BhvSingle{a} = cell(1, nDimBhv);  % d2 for each individual behavior latent
    d2NonBhvSingle{a} = cell(1, nDimNonBhv);  % d2 for each individual non-behavior latent
    startS{a} = nan(1, numWindows);
    
    % Calculate startS for all windows (same for all analyses)
    for w = 1:numWindows
        startIdx = (w - 1) * stepSamples + 1;
        startS{a}(w) = (startIdx + round(winSamples/2)-1) * optimalBinSize(a);
    end
    
    % For each number of dimensions from 1 to nDimBhv
    for nDim = 1:nDimBhv
        fprintf('    Processing %d dimensions...\n', nDim);
        
        % ========== Behavior-related latents ==========
        % Use first nDim behavior-related latents
        xPredBhv_nDim = xPred(:, 1:nDim);
        
        % Project back to neural space using learned mapping
        % Use only the first nDim rows of W_full (corresponding to first nDim latents)
        W_bhv_nDim = W_full(1:nDim, :);
        yPredBhv_nDim = xPredBhv_nDim * W_bhv_nDim;
        
        % Calculate population activity from reconstructed neural data
        popActivityBhv_nDim = round(sum(yPredBhv_nDim, 2));
        popActivityBhv{a}{nDim} = popActivityBhv_nDim;
        
        % Calculate d2 using sliding window
        d2Bhv_nDim = nan(1, numWindows);
        for w = 1:numWindows
            startIdx = (w - 1) * stepSamples + 1;
            endIdx = startIdx + winSamples - 1;
            wPopActivity = popActivityBhv_nDim(startIdx:endIdx);
            
            if analyzeD2
                if std(wPopActivity) > 0
                [varphi, ~] = myYuleWalker3(wPopActivity, pOrder);
                d2Bhv_nDim(w) = getFixedPointDistance2(pOrder, critType, varphi);
                end
            end
        end
        d2Bhv{a}{nDim} = d2Bhv_nDim;
        
        % ========== Non-behavior latents ==========
        if nDim <= (size(xPred, 2) - nDimBhv)
            % Use first nDim non-behavior latents
            xPredNonBhv_nDim = xPred(:, nDimBhv+1:nDimBhv+nDim);
            
            % Project back to neural space using learned mapping
            % Use rows nDimBhv+1 to nDimBhv+nDim of W_full
            W_nonBhv_nDim = W_full(nDimBhv+1:nDimBhv+nDim, :);
            yPredNonBhv_nDim = xPredNonBhv_nDim * W_nonBhv_nDim;
            
            % Calculate population activity
            popActivityNonBhv_nDim = round(sum(yPredNonBhv_nDim, 2));
            popActivityNonBhv{a}{nDim} = popActivityNonBhv_nDim;
            
            % Calculate d2 using sliding window
            d2NonBhv_nDim = nan(1, numWindows);
            for w = 1:numWindows
                startIdx = (w - 1) * stepSamples + 1;
                endIdx = startIdx + winSamples - 1;
                wPopActivity = popActivityNonBhv_nDim(startIdx:endIdx);
                
                if analyzeD2
                 if std(wPopActivity) > 0
                   [varphi, ~] = myYuleWalker3(wPopActivity, pOrder);
                    d2NonBhv_nDim(w) = getFixedPointDistance2(pOrder, critType, varphi);
                 end
                end
            end
            d2NonBhv{a}{nDim} = d2NonBhv_nDim;
        end
    end
    
    % ========== Step 5: Calculate d2 for each individual latent ==========
    fprintf('  Calculating d2 for individual latents...\n');
    
    % Behavior-related individual latents
    for dim = 1:nDimBhv
        fprintf('    Processing behavior latent %d/%d...\n', dim, nDimBhv);
        xPredBhv_single = xPred(:, dim);  % Single latent dimension
        
        % Calculate d2 directly from latent (no projection needed)
        d2Bhv_single = nan(1, numWindows);
        for w = 1:numWindows
            startIdx = (w - 1) * stepSamples + 1;
            endIdx = startIdx + winSamples - 1;
            wLatent = xPredBhv_single(startIdx:endIdx);
            
            if analyzeD2
                [varphi, ~] = myYuleWalker3(wLatent, pOrder);
                d2Bhv_single(w) = getFixedPointDistance2(pOrder, critType, varphi);
            end
        end
        d2BhvSingle{a}{dim} = d2Bhv_single;
    end
    
    % Non-behavior individual latents
    nNonBhvDims = size(xPred, 2) - nDimBhv;
    for dim = 1:min(nDimNonBhv, nNonBhvDims)
        fprintf('    Processing non-behavior latent %d/%d...\n', dim, nNonBhvDims);
        xPredNonBhv_single = xPred(:, nDimBhv + dim);  % Single latent dimension
        
        % Calculate d2 directly from latent (no projection needed)
        d2NonBhv_single = nan(1, numWindows);
        for w = 1:numWindows
            startIdx = (w - 1) * stepSamples + 1;
            endIdx = startIdx + winSamples - 1;
            wLatent = xPredNonBhv_single(startIdx:endIdx);
            
            if analyzeD2
                [varphi, ~] = myYuleWalker3(wLatent, pOrder);
                d2NonBhv_single(w) = getFixedPointDistance2(pOrder, critType, varphi);
            end
        end
        d2NonBhvSingle{a}{dim} = d2NonBhv_single;
    end
    
    fprintf('Area %s completed in %.1f minutes\n', areas{a}, toc/60);
end

% =============================    Save Results    =============================
results = struct();
results.dataType = dataType;
results.areas = areas;
results.d2Bhv = d2Bhv;
results.d2NonBhv = d2NonBhv;
results.d2BhvSingle = d2BhvSingle;
results.d2NonBhvSingle = d2NonBhvSingle;
results.startS = startS;
results.popActivityBhv = popActivityBhv;
results.popActivityNonBhv = popActivityNonBhv;
results.optimalBinSize = optimalBinSize;
results.optimalWindowSize = optimalWindowSize;
results.d2StepSize = d2StepSizeData;
results.d2WindowSize = d2WindowSizeData;
results.params.slidingWindowSize = slidingWindowSize;
results.params.psidInputType = psidInputType;
results.params.nDimBhv = nDimBhv;
results.params.nDimNonBhv = nDimNonBhv;
results.params.psidI = psidI;
results.params.pOrder = pOrder;
results.params.critType = critType;

save(resultsPath, 'results');
fprintf('Saved %s d2 PSID results to %s\n', dataType, resultsPath);

%% =============================    Plotting    =============================
if makePlots
    % Detect monitors and size figure to full screen (prefer second monitor if present)
    monitorPositions = get(0, 'MonitorPositions');
    monitorOne = monitorPositions(1, :);
    monitorTwo = monitorPositions(size(monitorPositions, 1), :);
    if size(monitorPositions, 1) >= 2
        targetPos = monitorTwo;
    else
        targetPos = monitorOne;
    end
    
    % Create plots
    if strcmp(dataType, 'reach')
        reachOnsetTimes = dataR.R(:,1) / 1000; % Convert from ms to seconds
    else
        reachOnsetTimes = [];
    end
    
    plot_psid_criticality_timeseries(startS, d2Bhv, d2NonBhv, d2BhvSingle, d2NonBhvSingle, areasToTest, areas, dataType, slidingWindowSize, saveDir, targetPos, nDimBhv, reachOnsetTimes, filenameSuffix);
end

fprintf('\n=== %s PSID Analysis Complete ===\n', dataType);

%% =============================    Function Definitions    =============================

function plot_psid_criticality_timeseries(startS, d2Bhv, d2NonBhv, d2BhvSingle, d2NonBhvSingle, areasToTest, areas, dataType, slidingWindowSize, saveDir, targetPos, nDimBhv, reachOnsetTimes, filenameSuffix)
% Plot time series of d2 criticality measures for PSID latents
% Creates separate figures for each brain area
% For each area: 2 plots (iterative projections and single latents)

% Colors for different numbers of dimensions
dimColors = lines(nDimBhv);
maxY = .08;
% Plot for each area separately
for idx = 1:length(areasToTest)
    a = areasToTest(idx);
    areaName = areas{a};
    
    % ========== Plot 1: Iterative 1:nDim projections ==========
    figure(900 + a); clf;
    set(gcf, 'Units', 'pixels');
    set(gcf, 'Position', targetPos);
    ha = tight_subplot(2, 1, [0.08 0.04], [0.08 0.1], [0.06 0.04]);
    
    % Top subplot: Behavior-related latents (iterative)
    axes(ha(1));
    hold on;
    ylim([0 maxY])    
    for nDim = 1:nDimBhv
        if ~isempty(d2Bhv{a}) && length(d2Bhv{a}) >= nDim && ~isempty(d2Bhv{a}{nDim})
            plot(startS{a}, d2Bhv{a}{nDim}, '-', 'Color', dimColors(nDim, :), ...
                'LineWidth', 2, 'DisplayName', sprintf('%d dims', nDim));
        end
    end
    
    % Add vertical lines at reach onsets (only for reach data)
    if ~isempty(reachOnsetTimes) && strcmp(dataType, 'reach')
        if ~isempty(startS{a})
            plotTimeRange = [startS{a}(1), startS{a}(end)];
            reachOnsetsInRange = reachOnsetTimes(reachOnsetTimes >= plotTimeRange(1) & reachOnsetTimes <= plotTimeRange(2));
            if ~isempty(reachOnsetsInRange)
                for i = 1:length(reachOnsetsInRange)
                    xline(reachOnsetsInRange(i), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, 'LineStyle', '--', 'Alpha', 0.7);
                end
            end
        end
    end
    
    if ~isempty(startS{a})
        xlim([startS{a}(1) startS{a}(end)]);
    end
    ylabel('d2', 'FontSize', 12);
    title(sprintf('%s - Behavior Latents (Iterative 1:%d dims)', areaName, nDimBhv), 'FontSize', 12);
    legend('Location', 'best', 'FontSize', 10);
    grid on;
    
    % Bottom subplot: Non-behavior latents (iterative)
    axes(ha(2));
    hold on;
    ylim([0 maxY])    
    for nDim = 1:nDimBhv
        if ~isempty(d2NonBhv{a}) && length(d2NonBhv{a}) >= nDim && ~isempty(d2NonBhv{a}{nDim})
            plot(startS{a}, d2NonBhv{a}{nDim}, '-', 'Color', dimColors(nDim, :), ...
                'LineWidth', 2, 'DisplayName', sprintf('%d dims', nDim));
        end
    end
    
    % Add vertical lines at reach onsets
    if ~isempty(reachOnsetTimes) && strcmp(dataType, 'reach')
        if ~isempty(startS{a})
            plotTimeRange = [startS{a}(1), startS{a}(end)];
            reachOnsetsInRange = reachOnsetTimes(reachOnsetTimes >= plotTimeRange(1) & reachOnsetTimes <= plotTimeRange(2));
            if ~isempty(reachOnsetsInRange)
                for i = 1:length(reachOnsetsInRange)
                    xline(reachOnsetsInRange(i), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, 'LineStyle', '--', 'Alpha', 0.7);
                end
            end
        end
    end
    
    if ~isempty(startS{a})
        xlim([startS{a}(1) startS{a}(end)]);
    end
    xlabel('Time (s)', 'FontSize', 12);
    ylabel('d2', 'FontSize', 12);
    title(sprintf('%s - Non-Behavior Latents (Iterative 1:%d dims)', areaName, nDimBhv), 'FontSize', 12);
    legend('Location', 'best', 'FontSize', 10);
    grid on;
    
    sgtitle(sprintf('%s - %s PSID d2 (Iterative Projections) - win=%gs', areaName, dataType, slidingWindowSize), 'FontSize', 14);
    
    % Save iterative projections plot
    saveFileIter = fullfile(saveDir, sprintf('criticality_%s_%s_psid%s_win%d_iterative.eps', areaName, dataType, filenameSuffix, slidingWindowSize));
    exportgraphics(gcf, saveFileIter, 'ContentType', 'vector');
    fprintf('Saved iterative projections plot for %s to: %s\n', areaName, saveFileIter);
    
    % ========== Plot 2: Single latents ==========
    figure(950 + a); clf;
    set(gcf, 'Units', 'pixels');
    set(gcf, 'Position', targetPos);
    ha = tight_subplot(2, 1, [0.08 0.04], [0.08 0.1], [0.06 0.04]);
    
    % Top subplot: Individual behavior latents
    axes(ha(1));
    hold on;
    ylim([0 maxY])    
    
    for dim = 1:nDimBhv
        if ~isempty(d2BhvSingle{a}) && length(d2BhvSingle{a}) >= dim && ~isempty(d2BhvSingle{a}{dim})
            plot(startS{a}, d2BhvSingle{a}{dim}, '-', 'Color', dimColors(dim, :), ...
                'LineWidth', 2, 'DisplayName', sprintf('Latent %d', dim));
        end
    end
    
    % Add vertical lines at reach onsets
    if ~isempty(reachOnsetTimes) && strcmp(dataType, 'reach')
        if ~isempty(startS{a})
            plotTimeRange = [startS{a}(1), startS{a}(end)];
            reachOnsetsInRange = reachOnsetTimes(reachOnsetTimes >= plotTimeRange(1) & reachOnsetTimes <= plotTimeRange(2));
            if ~isempty(reachOnsetsInRange)
                for i = 1:length(reachOnsetsInRange)
                    xline(reachOnsetsInRange(i), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, 'LineStyle', '--', 'Alpha', 0.7);
                end
            end
        end
    end
    
    if ~isempty(startS{a})
        xlim([startS{a}(1) startS{a}(end)]);
    end
    ylabel('d2', 'FontSize', 12);
    title(sprintf('%s - Individual Behavior Latents', areaName), 'FontSize', 12);
    legend('Location', 'best', 'FontSize', 10);
    grid on;
    
    % Bottom subplot: Individual non-behavior latents
    axes(ha(2));
    hold on;
    ylim([0 maxY])    
    
    % Determine how many non-behavior latents we have
    nNonBhvDims = 0;
    if ~isempty(d2NonBhvSingle{a})
        nNonBhvDims = length(d2NonBhvSingle{a});
    end
    
    for dim = 1:nNonBhvDims
        if ~isempty(d2NonBhvSingle{a}{dim})
            plot(startS{a}, d2NonBhvSingle{a}{dim}, '-', 'Color', dimColors(dim, :), ...
                'LineWidth', 2, 'DisplayName', sprintf('Latent %d', dim));
        end
    end
    
    % Add vertical lines at reach onsets
    if ~isempty(reachOnsetTimes) && strcmp(dataType, 'reach')
        if ~isempty(startS{a})
            plotTimeRange = [startS{a}(1), startS{a}(end)];
            reachOnsetsInRange = reachOnsetTimes(reachOnsetTimes >= plotTimeRange(1) & reachOnsetTimes <= plotTimeRange(2));
            if ~isempty(reachOnsetsInRange)
                for i = 1:length(reachOnsetsInRange)
                    xline(reachOnsetsInRange(i), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, 'LineStyle', '--', 'Alpha', 0.7);
                end
            end
        end
    end
    
    if ~isempty(startS{a})
        xlim([startS{a}(1) startS{a}(end)]);
    end
    xlabel('Time (s)', 'FontSize', 12);
    ylabel('d2', 'FontSize', 12);
    title(sprintf('%s - Individual Non-Behavior Latents', areaName), 'FontSize', 12);
    legend('Location', 'best', 'FontSize', 10);
    grid on;
    
    sgtitle(sprintf('%s - %s PSID d2 (Individual Latents) - win=%gs', areaName, dataType, slidingWindowSize), 'FontSize', 14);
    
    % Save single latents plot
    saveFileSingle = fullfile(saveDir, sprintf('criticality_%s_%s_psid%s_win%d_single.eps', areaName, dataType, filenameSuffix, slidingWindowSize));
    exportgraphics(gcf, saveFileSingle, 'ContentType', 'vector');
    fprintf('Saved single latents plot for %s to: %s\n', areaName, saveFileSingle);
end
end

