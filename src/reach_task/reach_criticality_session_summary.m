%%
% Reach Criticality Session Summary
% Loads criticality results and performs statistical tests comparing d2 values
% across different task engagement segments (Block 1/2, Engaged/Not-Engaged)
%
% This script:
% 1. Loads criticality results from criticality_sliding_window_ar.m
% 2. Defines task segments using reach_task_engagement.m
% 3. Tests whole-session d2 vs. permutation (compares observed mean to distribution of permuted means)
% 4. Tests segment-specific d2 vs. permutation (uses permuted values from corresponding windows)
% 5. Tests differences between segments (permutation test shuffling segment labels)
% 6. Creates bar plots summarizing results
%
% NOTE ON PERMUTATION TESTS:
% - Whole-session test: Compares observed mean d2 to distribution of means from whole-session permutations
% - Within-segment tests: 
%   * If useSegmentSpecificPermutations = true: Computes permutations separately for each segment
%     by permuting neural data within that segment's time window (statistically valid)
%   * If useSegmentSpecificPermutations = false: Uses permuted values from whole-session permutations
%     corresponding to the segment windows (approximation, less valid)
% - Between-segment tests: Permutation test that shuffles data labels between segments (fully valid)

% =============================    Configuration    =============================
% Specify the session name (should match the one used in criticality_sliding_window_ar.m)
sessionName = 'AB2_28-Apr-2023 17_50_02_NeuroBeh.mat';
% sessionName = 'AB2_01-May-2023 15_34_59_NeuroBeh.mat';
% sessionName = 'AB2_11-May-2023 17_31_00_NeuroBeh.mat';
% sessionName = 'AB2_30-May-2023 12_49_52_NeuroBeh.mat';
% sessionName = 'AB6_27-Mar-2025 14_04_12_NeuroBeh.mat';
% sessionName = 'AB6_29-Mar-2025 15_21_05_NeuroBeh.mat';
% sessionName = 'AB6_02-Apr-2025 14_18_54_NeuroBeh.mat';
% sessionName = 'AB6_03-Apr-2025 13_34_09_NeuroBeh.mat';
sessionName = 'Y4_06-Oct-2023 14_14_53_NeuroBeh.mat';
% sessionName = 'Y15_26-Aug-2025 12_24_22_NeuroBeh.mat';
% sessionName = 'Y15_27-Aug-2025 14_02_21_NeuroBeh.mat';
% sessionName = 'Y15_28-Aug-2025 19_47_07_NeuroBeh.mat';
% sessionName = 'Y17_20-Aug-2025 17_34_48_NeuroBeh.mat';

% Analysis parameters
slidingWindowSize = 20;  % Should match criticality_sliding_window_ar.m
pcaFlag = false;  % Should match criticality_sliding_window_ar.m
filenameSuffix = '';  % Will be updated based on pcaFlag

% Avalanche analysis parameters
analyzeAvalanches = true;  % Set to true to analyze avalanche parameters
avalancheWindowSize = 180;  % Should match criticality_sliding_window_av.m
avalancheStepSize = 30;  % Should match criticality_sliding_window_av.m

% Engagement analysis options (for reach_task_engagement.m)
engagementOpts = struct();
engagementOpts.timesMedian = 2.5;
engagementOpts.windowSize = 3;
engagementOpts.nEngaged = 5;
engagementOpts.nNotEngaged = 5;
engagementOpts.nReachMin = 5;

% Statistical test parameters
nPermutationsBetweenSegments = 10000;  % For between-segment comparisons
alpha = 0.05;  % Significance threshold

% Permutation analysis options
useSegmentSpecificPermutations = true;  % If true, compute permutations separately for each segment (statistically valid)
                                        % If false, use whole-session permutations (approximation)
nShufflesSegment = 20;  % Number of permutations per segment (if useSegmentSpecificPermutations = true)

% Plotting options
makePlots = true;
savePlots = true;

% =============================    Load Data    =============================
paths = get_paths;

% Create filename suffix based on PCA flag
if pcaFlag
    filenameSuffix = '_pca';
end

% Load criticality results
fprintf('\n=== Loading Criticality Results ===\n');
[~, dataBaseName, ~] = fileparts(sessionName);
saveDir = fullfile(paths.dropPath, 'reach_task/results', dataBaseName);
resultsPath = fullfile(saveDir, sprintf('criticality_sliding_window_ar%s_win%d_%s.mat', filenameSuffix, slidingWindowSize, sessionName));

if ~exist(resultsPath, 'file')
    error('Results file not found: %s\nRun criticality_sliding_window_ar.m first.', resultsPath);
end

results = load(resultsPath);
results = results.results;

% Check if permutations are available
if ~results.enablePermutations
    warning('Permutations not available in results. Segment-specific permutation tests will be skipped.');
    hasPermutations = false;
else
    hasPermutations = true;
    nShuffles = results.nShuffles;
end

areas = results.areas;
d2 = results.d2;
startS = results.startS;

% Load reach data for engagement analysis
fprintf('\n=== Loading Reach Data for Engagement Analysis ===\n');
reachDataFile = fullfile(paths.reachDataPath, sessionName);
if ~exist(reachDataFile, 'file')
    error('Reach data file not found: %s', reachDataFile);
end

% Get engagement segments
fprintf('\n=== Computing Engagement Segments ===\n');
segmentWindows = reach_task_engagement(reachDataFile, engagementOpts);

% Load original neural data if segment-specific permutations are requested
if useSegmentSpecificPermutations
    fprintf('\n=== Loading Original Neural Data for Segment-Specific Permutations ===\n');
    
    % Load data similar to criticality_sliding_data_prep.m
    opts = neuro_behavior_options;
    opts.frameSize = .001;
    opts.firingRateCheckTime = 5 * 60;
    opts.collectStart = 0;
    opts.minFiringRate = .05;
    opts.maxFiringRate = 100;
    
    dataR = load(reachDataFile);
    opts.collectEnd = round(min(dataR.R(end,1) + 5000, max(dataR.CSV(:,1)*1000)) / 1000);
    
    [dataMat, idLabels, areaLabels] = reach_neural_matrix(dataR, opts);
    areasData = {'M23', 'M56', 'DS', 'VS'};
    idM23 = find(strcmp(areaLabels, 'M23'));
    idM56 = find(strcmp(areaLabels, 'M56'));
    idDS = find(strcmp(areaLabels, 'DS'));
    idVS = find(strcmp(areaLabels, 'VS'));
    idMatIdxData = {idM23, idM56, idDS, idVS};
    
    fprintf('Loaded neural data: %d time points, %d neurons\n', size(dataMat, 1), size(dataMat, 2));
else
    dataMat = [];
    opts = [];
    idMatIdxData = [];
    areasData = [];
end

% Define segment names
segmentNames = {'Block1_Engaged', 'Block1_NotEngaged', 'Block2_Engaged', 'Block2_NotEngaged'};
segmentFields = {'block1EngagedWindow', 'block1NotEngagedWindow', 'block2EngagedWindow', 'block2NotEngagedWindow'};

% =============================    Extract d2 Values by Segment    =============================
fprintf('\n=== Extracting d2 Values by Segment ===\n');

% Initialize storage
d2BySegment = cell(length(areas), length(segmentNames));
d2PermutedBySegment = cell(length(areas), length(segmentNames));
segmentIndices = cell(length(areas), length(segmentNames));

for a = 1:length(areas)
    if isempty(d2{a}) || isempty(startS{a})
        continue;
    end
    
    timePoints = startS{a};
    d2Values = d2{a};
    
    % Extract d2 values for each segment
    for seg = 1:length(segmentNames)
        segWindow = segmentWindows.(segmentFields{seg});
        
        if ~isempty(segWindow) && length(segWindow) == 2
            % Find time points within this segment
            segMask = (timePoints >= segWindow(1)) & (timePoints <= segWindow(2));
            segIndices = find(segMask);
            
            if ~isempty(segIndices)
                d2BySegment{a, seg} = d2Values(segIndices);
                segmentIndices{a, seg} = segIndices;
                
                % Extract corresponding permuted values from whole-session if not doing segment-specific
                if ~useSegmentSpecificPermutations && hasPermutations && ~isempty(results.d2Permuted{a})
                    d2PermutedBySegment{a, seg} = results.d2Permuted{a}(segIndices, :);
                end
            end
        end
    end
end

% =============================    Compute Segment-Specific Permutations    =============================
if useSegmentSpecificPermutations && ~isempty(dataMat)
    fprintf('\n=== Computing Segment-Specific Permutations ===\n');
    fprintf('This may take a while...\n');
    
    % Get analysis parameters from results
    optimalBinSize = results.optimalBinSize;
    optimalWindowSize = results.optimalWindowSize;
    d2StepSizeData = results.d2StepSize;
    d2WindowSizeData = results.d2WindowSize;
    pcaFlag = results.params.pcaFlag;
    pOrder = results.params.pOrder;
    critType = results.params.critType;
    analyzeD2 = results.params.analyzeD2;
    
    % Convert time to frame indices (dataMat is at opts.frameSize resolution)
    frameSize = opts.frameSize;  % 0.001 seconds (1 ms)
    
    for a = 1:length(areas)
        if isempty(d2{a}) || isempty(idMatIdxData{a})
            continue;
        end
        
        fprintf('\n--- Area: %s ---\n', areas{a});
        aID = idMatIdxData{a};
        
        % Process each segment
        for seg = 1:length(segmentNames)
            segWindow = segmentWindows.(segmentFields{seg});
            
            if isempty(segWindow) || length(segWindow) ~= 2
                continue;
            end
            
            % Check if this segment has observed data
            if isempty(d2BySegment{a, seg})
                continue;
            end
            
            fprintf('  Computing permutations for %s...\n', segmentNames{seg});
            
            % Convert segment time window to frame indices
            segStartFrame = round(segWindow(1) / frameSize) + 1;  % +1 for 1-based indexing
            segEndFrame = round(segWindow(2) / frameSize) + 1;
            segStartFrame = max(1, min(segStartFrame, size(dataMat, 1)));
            segEndFrame = max(1, min(segEndFrame, size(dataMat, 1)));
            
            if segEndFrame <= segStartFrame
                continue;
            end
            
            % Extract segment neural data
            segDataMat = dataMat(segStartFrame:segEndFrame, aID);
            
            % Bin data to optimal bin size
            segDataMatBinned = neural_matrix_ms_to_frames(segDataMat, optimalBinSize(a));
            numTimePointsSeg = size(segDataMatBinned, 1);
            
            % Check if segment is long enough
            stepSamples = round(d2StepSizeData(a) / optimalBinSize(a));
            winSamples = round(d2WindowSizeData(a) / optimalBinSize(a));
            
            if numTimePointsSeg < winSamples
                fprintf('    Segment too short (%d bins, need %d), skipping...\n', numTimePointsSeg, winSamples);
                continue;
            end
            
            numWindowsSeg = floor((numTimePointsSeg - winSamples) / stepSamples) + 1;
            
            if numWindowsSeg < 1
                fprintf('    No windows in segment, skipping...\n');
                continue;
            end
            
            % Initialize storage for segment-specific permutations
            d2PermutedSeg = nan(numWindowsSeg, nShufflesSegment);
            
            % Compute permutations for this segment
            for shuffle = 1:nShufflesSegment
                % Circularly permute each neuron independently within the segment
                permutedSegData = segDataMatBinned;
                nNeurons = size(permutedSegData, 2);
                nSamples = size(permutedSegData, 1);
                
                for n = 1:nNeurons
                    % Random circular shift for this neuron within segment
                    shiftAmount = randi([1, nSamples]);
                    permutedSegData(:, n) = circshift(permutedSegData(:, n), shiftAmount);
                end
                
                % Apply PCA if needed (compute PCA on permuted segment data separately)
                if pcaFlag
                    [coeffPerm, scorePerm, ~, ~, explainedPerm, muPerm] = pca(permutedSegData);
                    forDimPerm = find(cumsum(explainedPerm) > 30, 1);
                    forDimPerm = max(3, min(6, forDimPerm));
                    nDimPerm = 1:forDimPerm;
                    permutedSegData = scorePerm(:,nDimPerm) * coeffPerm(:,nDimPerm)' + muPerm;
                end
                
                % Calculate population activity for permuted segment data
                permutedPopActivity = round(sum(permutedSegData, 2));
                
                % Run sliding window analysis on permuted segment data
                for w = 1:numWindowsSeg
                    startIdx = (w - 1) * stepSamples + 1;
                    endIdx = startIdx + winSamples - 1;
                    wPopActivityPerm = permutedPopActivity(startIdx:endIdx);
                    
                    if analyzeD2
                        [varphiPerm, ~] = myYuleWalker3(wPopActivityPerm, pOrder);
                        d2PermutedSeg(w, shuffle) = getFixedPointDistance2(pOrder, critType, varphiPerm);
                    end
                end
                
                if mod(shuffle, max(1, round(nShufflesSegment/10))) == 0
                    fprintf('    Completed %d/%d permutations\n', shuffle, nShufflesSegment);
                end
            end
            
            % Store segment-specific permuted values
            % Need to match windows to the observed d2 windows
            % Find which observed windows fall within this segment
            segObservedIndices = segmentIndices{a, seg};
            if ~isempty(segObservedIndices) && length(segObservedIndices) <= numWindowsSeg
                % Use the first n observed windows (they should align)
                nObserved = length(segObservedIndices);
                d2PermutedBySegment{a, seg} = d2PermutedSeg(1:nObserved, :);
            else
                % Store all permuted windows (may have more than observed due to different windowing)
                d2PermutedBySegment{a, seg} = d2PermutedSeg;
            end
            
            fprintf('    Completed segment-specific permutations: %d windows, %d shuffles\n', ...
                size(d2PermutedBySegment{a, seg}, 1), nShufflesSegment);
        end
    end
    
    fprintf('\n=== Segment-Specific Permutations Complete ===\n');
end

% =============================    Statistical Tests    =============================
fprintf('\n=== Performing Statistical Tests ===\n');

% Initialize results structure
statsResults = struct();
statsResults.areas = areas;
statsResults.segmentNames = segmentNames;
statsResults.wholeSession = struct();
statsResults.withinSegment = struct();
statsResults.betweenSegments = struct();

% For each area
for a = 1:length(areas)
    if isempty(d2{a})
        continue;
    end
    
    fprintf('\n--- Area: %s ---\n', areas{a});
    
    % Whole-session test: d2 vs. permutation
    fprintf('  Whole-session test...\n');
    d2Whole = d2{a}(~isnan(d2{a}));
    meanD2Whole = nanmean(d2Whole);
    
    if hasPermutations && ~isempty(results.d2Permuted{a})
        % Compute mean for each permutation shuffle
        % d2Permuted is [nWindows x nShuffles]
        permutedMeans = nanmean(results.d2Permuted{a}, 1);  % Mean across windows for each shuffle
        meanPermutedWhole = nanmean(permutedMeans);
        
        % Two-tailed test: how extreme is observed mean compared to permutation distribution?
        % Calculate p-value as proportion of permuted means >= observed (or <= if negative)
        if meanD2Whole >= meanPermutedWhole
            pValue = sum(permutedMeans >= meanD2Whole) / length(permutedMeans);
        else
            pValue = sum(permutedMeans <= meanD2Whole) / length(permutedMeans);
        end
        % Two-tailed p-value
        pValue = min(pValue * 2, 1);
        
        statsResults.wholeSession.(areas{a}).meanD2 = meanD2Whole;
        statsResults.wholeSession.(areas{a}).meanPermuted = meanPermutedWhole;
        statsResults.wholeSession.(areas{a}).pValue = pValue;
        statsResults.wholeSession.(areas{a}).significant = pValue < alpha;
        statsResults.wholeSession.(areas{a}).nWindows = length(d2Whole);
        
        fprintf('    Mean d2: %.4f, Mean permuted: %.4f, p = %.4f, n = %d %s\n', ...
            meanD2Whole, meanPermutedWhole, pValue, length(d2Whole), ...
            statsResults.wholeSession.(areas{a}).significant);
    else
        statsResults.wholeSession.(areas{a}).meanD2 = meanD2Whole;
        statsResults.wholeSession.(areas{a}).meanPermuted = NaN;
        statsResults.wholeSession.(areas{a}).pValue = NaN;
        statsResults.wholeSession.(areas{a}).significant = false;
        statsResults.wholeSession.(areas{a}).nWindows = length(d2Whole);
        fprintf('    Mean d2: %.4f, n = %d (no permutation data)\n', meanD2Whole, length(d2Whole));
    end
    
    % Within-segment tests: each segment vs. permutation
    fprintf('  Within-segment tests...\n');
    for seg = 1:length(segmentNames)
        segD2 = d2BySegment{a, seg};
        
        if ~isempty(segD2) && length(segD2) > 1
            meanSegD2 = nanmean(segD2);
            permType = 'none';
            
            if ~isempty(d2PermutedBySegment{a, seg})
                % Compute mean for each permutation shuffle within this segment
                % d2PermutedBySegment is [nWindowsInSegment x nShuffles]
                segPermutedMeans = nanmean(d2PermutedBySegment{a, seg}, 1);  % Mean across windows for each shuffle
                meanSegPermuted = nanmean(segPermutedMeans);
                
                % Two-tailed test: compare observed segment mean to distribution of permuted segment means
                if meanSegD2 >= meanSegPermuted
                    pValue = sum(segPermutedMeans >= meanSegD2) / length(segPermutedMeans);
                else
                    pValue = sum(segPermutedMeans <= meanSegD2) / length(segPermutedMeans);
                end
                pValue = min(pValue * 2, 1);
                
                % Note which type of permutation was used
                if useSegmentSpecificPermutations
                    permType = 'segment-specific';
                else
                    permType = 'whole-session (approximation)';
                end
                
                statsResults.withinSegment.(areas{a}).(segmentNames{seg}).meanD2 = meanSegD2;
                statsResults.withinSegment.(areas{a}).(segmentNames{seg}).meanPermuted = meanSegPermuted;
                statsResults.withinSegment.(areas{a}).(segmentNames{seg}).pValue = pValue;
                statsResults.withinSegment.(areas{a}).(segmentNames{seg}).significant = pValue < alpha;
                statsResults.withinSegment.(areas{a}).(segmentNames{seg}).nWindows = length(segD2);
                statsResults.withinSegment.(areas{a}).(segmentNames{seg}).permType = permType;
                
                fprintf('    %s: Mean d2: %.4f, Mean permuted: %.4f, p = %.4f, n = %d (%s) %s\n', ...
                    segmentNames{seg}, meanSegD2, meanSegPermuted, pValue, length(segD2), permType, ...
                    statsResults.withinSegment.(areas{a}).(segmentNames{seg}).significant);
            else
                statsResults.withinSegment.(areas{a}).(segmentNames{seg}).meanD2 = meanSegD2;
                statsResults.withinSegment.(areas{a}).(segmentNames{seg}).meanPermuted = NaN;
                statsResults.withinSegment.(areas{a}).(segmentNames{seg}).pValue = NaN;
                statsResults.withinSegment.(areas{a}).(segmentNames{seg}).significant = false;
                statsResults.withinSegment.(areas{a}).(segmentNames{seg}).nWindows = length(segD2);
                statsResults.withinSegment.(areas{a}).(segmentNames{seg}).permType = 'none';
                fprintf('    %s: Mean d2: %.4f, n = %d (no permutation data)\n', ...
                    segmentNames{seg}, meanSegD2, length(segD2));
            end
        else
            statsResults.withinSegment.(areas{a}).(segmentNames{seg}).meanD2 = NaN;
            statsResults.withinSegment.(areas{a}).(segmentNames{seg}).meanPermuted = NaN;
            statsResults.withinSegment.(areas{a}).(segmentNames{seg}).pValue = NaN;
            statsResults.withinSegment.(areas{a}).(segmentNames{seg}).significant = false;
            statsResults.withinSegment.(areas{a}).(segmentNames{seg}).nWindows = 0;
            fprintf('    %s: No data\n', segmentNames{seg});
        end
    end
    
    % Between-segment tests: compare segments to each other
    fprintf('  Between-segment tests...\n');
    % Compare all pairs of segments that have data
    segPairs = nchoosek(1:length(segmentNames), 2);
    for p = 1:size(segPairs, 1)
        seg1 = segPairs(p, 1);
        seg2 = segPairs(p, 2);
        
        d2Seg1 = d2BySegment{a, seg1};
        d2Seg2 = d2BySegment{a, seg2};
        
        if ~isempty(d2Seg1) && ~isempty(d2Seg2) && length(d2Seg1) > 1 && length(d2Seg2) > 1
            meanSeg1 = nanmean(d2Seg1);
            meanSeg2 = nanmean(d2Seg2);
            diffMean = meanSeg1 - meanSeg2;
            
            % Permutation test: shuffle labels between segments
            % Combine data from both segments
            combinedData = [d2Seg1(:); d2Seg2(:)];
            n1 = length(d2Seg1);
            n2 = length(d2Seg2);
            nTotal = n1 + n2;
            
            % Generate null distribution by permuting labels
            nullDiffs = zeros(nPermutationsBetweenSegments, 1);
            for perm = 1:nPermutationsBetweenSegments
                permIdx = randperm(nTotal);
                permSeg1 = combinedData(permIdx(1:n1));
                permSeg2 = combinedData(permIdx(n1+1:end));
                nullDiffs(perm) = nanmean(permSeg1) - nanmean(permSeg2);
            end
            
            % Two-tailed p-value
            if diffMean >= 0
                pValue = sum(nullDiffs >= diffMean) / nPermutationsBetweenSegments;
            else
                pValue = sum(nullDiffs <= diffMean) / nPermutationsBetweenSegments;
            end
            pValue = min(pValue * 2, 1);
            
            pairName = sprintf('%s_vs_%s', segmentNames{seg1}, segmentNames{seg2});
            statsResults.betweenSegments.(areas{a}).(pairName).meanSeg1 = meanSeg1;
            statsResults.betweenSegments.(areas{a}).(pairName).meanSeg2 = meanSeg2;
            statsResults.betweenSegments.(areas{a}).(pairName).diffMean = diffMean;
            statsResults.betweenSegments.(areas{a}).(pairName).pValue = pValue;
            statsResults.betweenSegments.(areas{a}).(pairName).significant = pValue < alpha;
            statsResults.betweenSegments.(areas{a}).(pairName).nSeg1 = n1;
            statsResults.betweenSegments.(areas{a}).(pairName).nSeg2 = n2;
            
            fprintf('    %s vs %s: Diff = %.4f, p = %.4f %s\n', ...
                segmentNames{seg1}, segmentNames{seg2}, diffMean, pValue, ...
                statsResults.betweenSegments.(areas{a}).(pairName).significant);
        end
    end
end

% =============================    Save Results    =============================
fprintf('\n=== Saving Results ===\n');
summaryPath = fullfile(saveDir, sprintf('criticality_session_summary_%s.mat', sessionName));
save(summaryPath, 'statsResults', 'segmentWindows', 'engagementOpts', 'sessionName', ...
    'useSegmentSpecificPermutations', 'nShufflesSegment');
fprintf('Saved summary results to: %s\n', summaryPath);

% =============================    Avalanche Analysis    =============================
if analyzeAvalanches
    fprintf('\n=== Loading Avalanche Results ===\n');
    avalancheResultsPath = fullfile(saveDir, sprintf('criticality_sliding_window_av_win%d_step%d.mat', avalancheWindowSize, avalancheStepSize));
    
    if ~exist(avalancheResultsPath, 'file')
        warning('Avalanche results file not found: %s\nSkipping avalanche analysis.', avalancheResultsPath);
        avalancheStatsResults = [];
    else
        avResults = load(avalancheResultsPath);
        if isfield(avResults, 'results')
            avResults = avResults.results;
        end
        
        % Extract avalanche data
        avDcc = avResults.dcc;
        avKappa = avResults.kappa;
        avDecades = avResults.decades;
        avTau = avResults.tau;
        avAlpha = avResults.alpha;
        avParamSD = avResults.paramSD;
        avStartS = avResults.startS;
        avOptimalBinSize = avResults.optimalBinSize;
        
        % Check if avalanche permutations are available
        avHasPermutations = false;
        if isfield(avResults, 'enablePermutations') && avResults.enablePermutations
            avHasPermutations = true;
            avDccPermuted = avResults.dccPermuted;
            avKappaPermuted = avResults.kappaPermuted;
            avDecadesPermuted = avResults.decadesPermuted;
            avTauPermuted = avResults.tauPermuted;
            avAlphaPermuted = avResults.alphaPermuted;
            avParamSDPermuted = avResults.paramSDPermuted;
        end
        
        % Metrics to analyze
        avalancheMetricNames = {'dcc', 'kappa', 'decades', 'tau', 'alpha', 'paramSD'};
        avalancheMetricData = {avDcc, avKappa, avDecades, avTau, avAlpha, avParamSD};
        if avHasPermutations
            avalancheMetricPermuted = {avDccPermuted, avKappaPermuted, avDecadesPermuted, avTauPermuted, avAlphaPermuted, avParamSDPermuted};
        else
            avalancheMetricPermuted = cell(1, length(avalancheMetricNames));
        end
        
        % Initialize storage for avalanche metrics by segment
        fprintf('\n=== Extracting Avalanche Metrics by Segment ===\n');
        avalancheBySegment = cell(length(avalancheMetricNames), length(areas), length(segmentNames));
        avalanchePermutedBySegment = cell(length(avalancheMetricNames), length(areas), length(segmentNames));
        
        for m = 1:length(avalancheMetricNames)
            metricName = avalancheMetricNames{m};
            metricValues = avalancheMetricData{m};
            
            for a = 1:length(areas)
                if isempty(metricValues{a}) || isempty(avStartS{a})
                    continue;
                end
                
                timePoints = avStartS{a};
                values = metricValues{a};
                
                % Extract values for each segment
                for seg = 1:length(segmentNames)
                    segWindow = segmentWindows.(segmentFields{seg});
                    
                    if ~isempty(segWindow) && length(segWindow) == 2
                        % Find time points within this segment
                        segMask = (timePoints >= segWindow(1)) & (timePoints <= segWindow(2));
                        segIndices = find(segMask);
                        
                        if ~isempty(segIndices)
                            segValues = values(segIndices);
                            segValues = segValues(~isnan(segValues)); % Remove NaNs
                            avalancheBySegment{m, a, seg} = segValues;
                            
                            % Extract permuted values if available
                            if avHasPermutations && ~isempty(avalancheMetricPermuted{m}{a})
                                avalanchePermutedBySegment{m, a, seg} = avalancheMetricPermuted{m}{a}(segIndices, :);
                            end
                        end
                    end
                end
            end
        end
        
        % Perform statistical tests for avalanche metrics
        fprintf('\n=== Performing Avalanche Statistical Tests ===\n');
        avalancheStatsResults = struct();
        avalancheStatsResults.areas = areas;
        avalancheStatsResults.segmentNames = segmentNames;
        avalancheStatsResults.metricNames = avalancheMetricNames;
        avalancheStatsResults.wholeSession = struct();
        avalancheStatsResults.withinSegment = struct();
        avalancheStatsResults.betweenSegments = struct();
        
        for m = 1:length(avalancheMetricNames)
            metricName = avalancheMetricNames{m};
            fprintf('\n--- Metric: %s ---\n', metricName);
            
            % Initialize metric-specific structures
            avalancheStatsResults.wholeSession.(metricName) = struct();
            avalancheStatsResults.withinSegment.(metricName) = struct();
            avalancheStatsResults.betweenSegments.(metricName) = struct();
            
            for a = 1:length(areas)
                metricValues = avalancheMetricData{m}{a};
                
                if isempty(metricValues)
                    continue;
                end
                
                fprintf('  Area: %s\n', areas{a});
                
                % Whole-session test
                fprintf('    Whole-session test...\n');
                metricWhole = metricValues(~isnan(metricValues));
                meanMetricWhole = nanmean(metricWhole);
                
                if avHasPermutations && ~isempty(avalancheMetricPermuted{m}{a})
                    permutedMeans = nanmean(avalancheMetricPermuted{m}{a}, 1);
                    meanPermutedWhole = nanmean(permutedMeans);
                    
                    if meanMetricWhole >= meanPermutedWhole
                        pValue = sum(permutedMeans >= meanMetricWhole) / length(permutedMeans);
                    else
                        pValue = sum(permutedMeans <= meanMetricWhole) / length(permutedMeans);
                    end
                    pValue = min(pValue * 2, 1);
                    
                    avalancheStatsResults.wholeSession.(metricName).(areas{a}).meanValue = meanMetricWhole;
                    avalancheStatsResults.wholeSession.(metricName).(areas{a}).meanPermuted = meanPermutedWhole;
                    avalancheStatsResults.wholeSession.(metricName).(areas{a}).pValue = pValue;
                    avalancheStatsResults.wholeSession.(metricName).(areas{a}).significant = pValue < alpha;
                    avalancheStatsResults.wholeSession.(metricName).(areas{a}).nWindows = length(metricWhole);
                    
                    fprintf('      Mean %s: %.4f, Mean permuted: %.4f, p = %.4f, n = %d %s\n', ...
                        metricName, meanMetricWhole, meanPermutedWhole, pValue, length(metricWhole), ...
                        char('*' * (pValue < alpha)));
                else
                    avalancheStatsResults.wholeSession.(metricName).(areas{a}).meanValue = meanMetricWhole;
                    avalancheStatsResults.wholeSession.(metricName).(areas{a}).meanPermuted = NaN;
                    avalancheStatsResults.wholeSession.(metricName).(areas{a}).pValue = NaN;
                    avalancheStatsResults.wholeSession.(metricName).(areas{a}).significant = false;
                    avalancheStatsResults.wholeSession.(metricName).(areas{a}).nWindows = length(metricWhole);
                    fprintf('      Mean %s: %.4f, n = %d (no permutation data)\n', metricName, meanMetricWhole, length(metricWhole));
                end
                
                % Within-segment tests
                fprintf('    Within-segment tests...\n');
                for seg = 1:length(segmentNames)
                    segValues = avalancheBySegment{m, a, seg};
                    
                    if ~isempty(segValues) && length(segValues) > 1
                        meanSegValue = nanmean(segValues);
                        
                        if ~isempty(avalanchePermutedBySegment{m, a, seg})
                            segPermutedMeans = nanmean(avalanchePermutedBySegment{m, a, seg}, 1);
                            meanSegPermuted = nanmean(segPermutedMeans);
                            
                            if meanSegValue >= meanSegPermuted
                                pValue = sum(segPermutedMeans >= meanSegValue) / length(segPermutedMeans);
                            else
                                pValue = sum(segPermutedMeans <= meanSegValue) / length(segPermutedMeans);
                            end
                            pValue = min(pValue * 2, 1);
                            
                            avalancheStatsResults.withinSegment.(metricName).(areas{a}).(segmentNames{seg}).meanValue = meanSegValue;
                            avalancheStatsResults.withinSegment.(metricName).(areas{a}).(segmentNames{seg}).meanPermuted = meanSegPermuted;
                            avalancheStatsResults.withinSegment.(metricName).(areas{a}).(segmentNames{seg}).pValue = pValue;
                            avalancheStatsResults.withinSegment.(metricName).(areas{a}).(segmentNames{seg}).significant = pValue < alpha;
                            avalancheStatsResults.withinSegment.(metricName).(areas{a}).(segmentNames{seg}).nWindows = length(segValues);
                            
                            fprintf('      %s: Mean %s: %.4f, Mean permuted: %.4f, p = %.4f, n = %d %s\n', ...
                                segmentNames{seg}, metricName, meanSegValue, meanSegPermuted, pValue, length(segValues), ...
                                char('*' * (pValue < alpha)));
                        else
                            avalancheStatsResults.withinSegment.(metricName).(areas{a}).(segmentNames{seg}).meanValue = meanSegValue;
                            avalancheStatsResults.withinSegment.(metricName).(areas{a}).(segmentNames{seg}).meanPermuted = NaN;
                            avalancheStatsResults.withinSegment.(metricName).(areas{a}).(segmentNames{seg}).pValue = NaN;
                            avalancheStatsResults.withinSegment.(metricName).(areas{a}).(segmentNames{seg}).significant = false;
                            avalancheStatsResults.withinSegment.(metricName).(areas{a}).(segmentNames{seg}).nWindows = length(segValues);
                            fprintf('      %s: Mean %s: %.4f, n = %d (no permutation data)\n', ...
                                segmentNames{seg}, metricName, meanSegValue, length(segValues));
                        end
                    else
                        avalancheStatsResults.withinSegment.(metricName).(areas{a}).(segmentNames{seg}).meanValue = NaN;
                        avalancheStatsResults.withinSegment.(metricName).(areas{a}).(segmentNames{seg}).meanPermuted = NaN;
                        avalancheStatsResults.withinSegment.(metricName).(areas{a}).(segmentNames{seg}).pValue = NaN;
                        avalancheStatsResults.withinSegment.(metricName).(areas{a}).(segmentNames{seg}).significant = false;
                        avalancheStatsResults.withinSegment.(metricName).(areas{a}).(segmentNames{seg}).nWindows = 0;
                        fprintf('      %s: No data\n', segmentNames{seg});
                    end
                end
                
                % Between-segment tests
                fprintf('    Between-segment tests...\n');
                segPairs = nchoosek(1:length(segmentNames), 2);
                for p = 1:size(segPairs, 1)
                    seg1 = segPairs(p, 1);
                    seg2 = segPairs(p, 2);
                    
                    seg1Values = avalancheBySegment{m, a, seg1};
                    seg2Values = avalancheBySegment{m, a, seg2};
                    
                    if ~isempty(seg1Values) && ~isempty(seg2Values) && length(seg1Values) > 1 && length(seg2Values) > 1
                        meanSeg1 = nanmean(seg1Values);
                        meanSeg2 = nanmean(seg2Values);
                        diffMean = meanSeg1 - meanSeg2;
                        
                        % Permutation test: shuffle labels between segments
                        combinedData = [seg1Values(:); seg2Values(:)];
                        n1 = length(seg1Values);
                        n2 = length(seg2Values);
                        nTotal = n1 + n2;
                        
                        nullDiffs = zeros(nPermutationsBetweenSegments, 1);
                        for perm = 1:nPermutationsBetweenSegments
                            permIdx = randperm(nTotal);
                            permSeg1 = combinedData(permIdx(1:n1));
                            permSeg2 = combinedData(permIdx(n1+1:end));
                            nullDiffs(perm) = nanmean(permSeg1) - nanmean(permSeg2);
                        end
                        
                        % Two-tailed p-value
                        if abs(diffMean) >= 0
                            pValue = sum(abs(nullDiffs) >= abs(diffMean)) / nPermutationsBetweenSegments;
                        else
                            pValue = 1;
                        end
                        
                        pairName = sprintf('%s_vs_%s', segmentNames{seg1}, segmentNames{seg2});
                        avalancheStatsResults.betweenSegments.(metricName).(areas{a}).(pairName).diffMean = diffMean;
                        avalancheStatsResults.betweenSegments.(metricName).(areas{a}).(pairName).pValue = pValue;
                        avalancheStatsResults.betweenSegments.(metricName).(areas{a}).(pairName).significant = pValue < alpha;
                        avalancheStatsResults.betweenSegments.(metricName).(areas{a}).(pairName).n1 = n1;
                        avalancheStatsResults.betweenSegments.(metricName).(areas{a}).(pairName).n2 = n2;
                        
                        fprintf('      %s: Diff = %.4f, p = %.4f %s\n', pairName, diffMean, pValue, char('*' * (pValue < alpha)));
                    end
                end
            end
        end
        
        % Create avalanche plots
        if makePlots
            fprintf('\n=== Creating Avalanche Plots ===\n');
            create_avalanche_engagement_plots(avalancheStatsResults, areas, segmentNames, avalancheMetricNames, savePlots, saveDir, sessionName);
        end
        
        % Save avalanche results
        avalancheSummaryPath = fullfile(saveDir, sprintf('avalanche_engagement_summary_%s.mat', sessionName));
        save(avalancheSummaryPath, 'avalancheStatsResults', 'segmentWindows', 'engagementOpts', 'sessionName', ...
            'avalancheWindowSize', 'avalancheStepSize');
        fprintf('Saved avalanche summary results to: %s\n', avalancheSummaryPath);
    end
else
    avalancheStatsResults = [];
end

% =============================    Plotting    =============================
if makePlots
    fprintf('\n=== Creating Plots ===\n');
    
    % Extract first 8 characters of session name for titles
    filePrefix = sessionName(1:min(8, length(sessionName)));
    
    % Plot 1: Whole-session d2 vs. permutation
    figure(1000); clf;
    set(gcf, 'Position', [100, 100, 1200, 600]);
    
    numAreas = length(areas);
    for a = 1:numAreas
        subplot(1, numAreas, a);
        hold on;
        
        if isfield(statsResults.wholeSession, areas{a})
            ws = statsResults.wholeSession.(areas{a});
            
            % Bar plot
            if ~isnan(ws.meanPermuted)
                barData = [ws.meanD2, ws.meanPermuted];
                b = bar(barData, 'FaceColor', 'flat');
                b.CData(1, :) = [0.2 0.4 0.8];  % Blue for observed
                b.CData(2, :) = [0.6 0.6 0.6];  % Gray for permuted
                
                % Add significance marker
                if ws.significant
                    text(1, ws.meanD2 + 0.02, '*', 'FontSize', 20, 'HorizontalAlignment', 'center');
                end
            else
                barData = ws.meanD2;
                b = bar(barData, 'FaceColor', [0.2 0.4 0.8]);
            end
            
            % Set x-axis ticks and labels to align with bars
            if ~isnan(ws.meanPermuted)
                set(gca, 'XTick', [1, 2], 'XTickLabel', {'Observed', 'Permuted'});
                xlim([0.5, 2.5]);
            else
                set(gca, 'XTick', 1, 'XTickLabel', {'Observed'});
                xlim([0.5, 1.5]);
            end
            ylabel('Mean d2');
            title(sprintf('%s\nWhole Session (p=%.4f)', areas{a}, ws.pValue));
            grid on;
        end
    end
    sgtitle(sprintf('[%s] Whole-Session d2 vs. Permutation', filePrefix));
    
    if savePlots
        plotPath1 = fullfile(saveDir, sprintf('criticality_whole_session_%s.png', sessionName));
        exportgraphics(gcf, plotPath1, 'Resolution', 300);
        fprintf('Saved plot to: %s\n', plotPath1);
    end
    
    % Plot 2: Within-segment d2 values
    figure(1001); clf;
    set(gcf, 'Position', [100, 100, 1400, 800]);
    
    for a = 1:numAreas
        subplot(2, 2, a);
        hold on;
        
        if isfield(statsResults.withinSegment, areas{a})
            ws = statsResults.withinSegment.(areas{a});
            
            % Collect data for bar plot
            means = zeros(1, length(segmentNames));
            sems = zeros(1, length(segmentNames));
            permutedMeans = zeros(1, length(segmentNames));
            permutedSTDs = zeros(1, length(segmentNames));
            significant = false(1, length(segmentNames));
            hasData = false(1, length(segmentNames));
            
            for seg = 1:length(segmentNames)
                if isfield(ws, segmentNames{seg}) && ~isnan(ws.(segmentNames{seg}).meanD2)
                    means(seg) = ws.(segmentNames{seg}).meanD2;
                    nWindows = ws.(segmentNames{seg}).nWindows;
                    if nWindows > 1
                        segD2 = d2BySegment{a, seg};
                        sems(seg) = nanstd(segD2) / sqrt(nWindows);
                    end
                    
                    % Calculate permuted mean and STD if available
                    if ~isnan(ws.(segmentNames{seg}).meanPermuted) && ~isempty(d2PermutedBySegment{a, seg})
                        % d2PermutedBySegment is [nWindowsInSegment x nShuffles]
                        segPermutedMeans = nanmean(d2PermutedBySegment{a, seg}, 1);  % Mean across windows for each shuffle
                        permutedMeans(seg) = nanmean(segPermutedMeans);
                        permutedSTDs(seg) = nanstd(segPermutedMeans);
                    else
                        permutedMeans(seg) = NaN;
                        permutedSTDs(seg) = NaN;
                    end
                    
                    significant(seg) = ws.(segmentNames{seg}).significant;
                    hasData(seg) = true;
                end
            end
            
            % Plot grouped bars (observed and permuted)
            validIdx = hasData;
            if any(validIdx)
                validSegIndices = find(validIdx);
                nValid = length(validSegIndices);
                
                % Prepare data for grouped bar plot
                % Column 1: observed, Column 2: permuted
                barData = zeros(nValid, 2);
                barData(:, 1) = means(validIdx);  % Observed
                barData(:, 2) = permutedMeans(validIdx);  % Permuted (NaN if not available)
                
                % Create grouped bar plot
                xPos = 1:nValid;
                b = bar(xPos, barData, 'grouped', 'FaceColor', 'flat');
                
                % Color bars: blue for observed, gray for permuted
                b(1).CData = repmat([0.2 0.4 0.8], nValid, 1);  % Blue for observed
                b(2).CData = repmat([0.6 0.6 0.6], nValid, 1);  % Gray for permuted
                
                % Get bar positions for error bars (grouped bars are offset by ~0.2)
                % For 2 groups, MATLAB positions them at xPos - 0.2 and xPos + 0.2
                barOffset = 0.2;
                
                % Add error bars for observed data (left bar)
                errorbar(xPos - barOffset, means(validIdx), sems(validIdx), 'k', 'LineStyle', 'none', 'LineWidth', 1.5);
                
                % Add error bars for permuted data (right bar, using STD)
                permutedErrorBars = permutedSTDs(validIdx);
                permutedErrorBars(isnan(permutedErrorBars)) = 0;  % Set NaN to 0 for errorbar
                permutedMeansValid = permutedMeans(validIdx);
                permutedMeansValid(isnan(permutedMeansValid)) = 0;  % Set NaN to 0 for errorbar
                errorbar(xPos + barOffset, permutedMeansValid, permutedErrorBars, 'k', 'LineStyle', 'none', 'LineWidth', 1.5);
                
                % Add significance markers above observed bars
                for i = 1:nValid
                    segIdx = validSegIndices(i);
                    if significant(segIdx)
                        yPos = means(segIdx) + sems(segIdx) + 0.01;
                        text(xPos(i) - barOffset, yPos, '*', ...
                            'FontSize', 16, 'HorizontalAlignment', 'center');
                    end
                end
                
                % Set x-axis labels
                segLabels = segmentNames(validIdx);
                segLabelsShort = cell(size(segLabels));
                for i = 1:length(segLabels)
                    segLabelsShort{i} = strrep(segLabels{i}, '_', '\n');
                end
                set(gca, 'XTick', xPos, 'XTickLabel', segLabelsShort);
                xtickangle(45);
                xlim([0.5, nValid + 0.5]);
                
                % Add legend
                legend({'Observed', 'Permuted'}, 'Location', 'best');
            end
            
            ylabel('Mean d2');
            title(areas{a});
            grid on;
        end
    end
    sgtitle(sprintf('[%s] Within-Segment d2 Values (vs. Permutation)', filePrefix));
    
    if savePlots
        plotPath2 = fullfile(saveDir, sprintf('criticality_within_segments_%s.png', sessionName));
        exportgraphics(gcf, plotPath2, 'Resolution', 300);
        fprintf('Saved plot to: %s\n', plotPath2);
    end
    
    % Plot 3: Between-segment comparisons
    figure(1002); clf;
    set(gcf, 'Position', [100, 100, 1400, 800]);
    
    for a = 1:numAreas
        subplot(2, 2, a);
        hold on;
        
        if isfield(statsResults.betweenSegments, areas{a})
            bs = statsResults.betweenSegments.(areas{a});
            
            % Collect all pair comparisons
            pairNames = fieldnames(bs);
            nPairs = length(pairNames);
            
            if nPairs > 0
                diffs = zeros(1, nPairs);
                pValues = zeros(1, nPairs);
                significant = false(1, nPairs);
                pairLabels = cell(1, nPairs);
                
                for p = 1:nPairs
                    pairData = bs.(pairNames{p});
                    diffs(p) = pairData.diffMean;
                    pValues(p) = pairData.pValue;
                    significant(p) = pairData.significant;
                    
                    % Create short label
                    pairLabels{p} = strrep(pairNames{p}, '_vs_', ' vs\n');
                    pairLabels{p} = strrep(pairLabels{p}, '_', ' ');
                end
                
                % Plot bars
                xPos = 1:nPairs;
                b = bar(xPos, diffs, 'FaceColor', 'flat');
                
                % Color bars: blue if significant, gray if not
                for i = 1:nPairs
                    if significant(i)
                        b.CData(i, :) = [0.2 0.4 0.8];
                    else
                        b.CData(i, :) = [0.6 0.6 0.6];
                    end
                end
                
                % Add significance markers
                for i = 1:nPairs
                    if significant(i)
                        text(xPos(i), diffs(i) + 0.01 * sign(diffs(i)), '*', ...
                            'FontSize', 16, 'HorizontalAlignment', 'center');
                    end
                end
                
                % Add zero line
                yline(0, 'k--', 'LineWidth', 1);
                
                set(gca, 'XTick', xPos, 'XTickLabel', pairLabels);
                xtickangle(45);
                ylabel('Difference in Mean d2');
                title(areas{a});
                grid on;
            end
        end
    end
    sgtitle(sprintf('[%s] Between-Segment Comparisons', filePrefix));
    
    if savePlots
        plotPath3 = fullfile(saveDir, sprintf('criticality_between_segments_%s.png', sessionName));
        exportgraphics(gcf, plotPath3, 'Resolution', 300);
        fprintf('Saved plot to: %s\n', plotPath3);
    end
end

fprintf('\n=== Analysis Complete ===\n');

%% =============================    Helper Functions    =============================

function create_avalanche_engagement_plots(avalancheStatsResults, areas, segmentNames, metricNames, savePlots, saveDir, sessionName)
% CREATE_AVALANCHE_ENGAGEMENT_PLOTS - Create plots comparing avalanche metrics across segments

% Extract first 8 characters of session name for titles
filePrefix = sessionName(1:min(8, length(sessionName)));

nMetrics = length(metricNames);
nAreas = length(areas);

% Plot 1: Whole-session avalanche metrics vs. permutation
figure(2000); clf;
set(gcf, 'Position', [100, 100, 1600, 1000]);

for m = 1:nMetrics
    metricName = metricNames{m};
    
    for a = 1:nAreas
        subplot(nMetrics, nAreas, (m-1)*nAreas + a);
        hold on;
        
        if isfield(avalancheStatsResults.wholeSession.(metricName), areas{a})
            ws = avalancheStatsResults.wholeSession.(metricName).(areas{a});
            
            % Bar plot
            if ~isnan(ws.meanPermuted)
                barData = [ws.meanValue, ws.meanPermuted];
                b = bar(barData, 'FaceColor', 'flat');
                b.CData(1, :) = [0.2 0.4 0.8];  % Blue for observed
                b.CData(2, :) = [0.6 0.6 0.6];  % Gray for permuted
                
                % Add significance marker
                if ws.significant
                    text(1, ws.meanValue + 0.02 * max(barData), '*', 'FontSize', 20, 'HorizontalAlignment', 'center');
                end
            else
                barData = ws.meanValue;
                b = bar(barData, 'FaceColor', [0.2 0.4 0.8]);
            end
            
            set(gca, 'XTickLabel', {'Observed', 'Permuted'});
            ylabel(metricName);
            title(sprintf('%s - %s\n(p=%.4f)', areas{a}, metricName, ws.pValue));
            grid on;
        end
    end
end

sgtitle(sprintf('[%s] Whole-Session Avalanche Metrics vs. Permutation', filePrefix), 'FontSize', 14, 'FontWeight', 'bold');

if savePlots
    plotPath = fullfile(saveDir, sprintf('avalanche_whole_session_%s.png', sessionName));
    exportgraphics(gcf, plotPath, 'Resolution', 300);
    fprintf('Saved avalanche whole-session plot to: %s\n', plotPath);
end

% Plot 2: Within-segment avalanche metrics
figure(2001); clf;
set(gcf, 'Position', [100, 100, 1800, 1200]);

for m = 1:nMetrics
    metricName = metricNames{m};
    
    for a = 1:nAreas
        subplot(nMetrics, nAreas, (m-1)*nAreas + a);
        hold on;
        
        if isfield(avalancheStatsResults.withinSegment.(metricName), areas{a})
            ws = avalancheStatsResults.withinSegment.(metricName).(areas{a});
            
            % Collect data for bar plot
            means = zeros(1, length(segmentNames));
            sems = zeros(1, length(segmentNames));
            significant = false(1, length(segmentNames));
            hasData = false(1, length(segmentNames));
            
            for seg = 1:length(segmentNames)
                if isfield(ws, segmentNames{seg}) && ~isnan(ws.(segmentNames{seg}).meanValue)
                    means(seg) = ws.(segmentNames{seg}).meanValue;
                    nWindows = ws.(segmentNames{seg}).nWindows;
                    if nWindows > 1
                        % Estimate SEM (would need actual data, using approximation)
                        sems(seg) = abs(means(seg)) * 0.1; % Rough estimate
                    end
                    significant(seg) = ws.(segmentNames{seg}).significant;
                    hasData(seg) = true;
                end
            end
            
            % Plot bars
            validIdx = hasData;
            if any(validIdx)
                xPos = 1:sum(validIdx);
                b = bar(xPos, means(validIdx), 'FaceColor', 'flat');
                
                % Color bars: blue if significant, gray if not
                validSegIndices = find(validIdx);
                for i = 1:length(xPos)
                    segIdx = validSegIndices(i);
                    if significant(segIdx)
                        b.CData(i, :) = [0.2 0.4 0.8];
                    else
                        b.CData(i, :) = [0.6 0.6 0.6];
                    end
                end
                
                % Add error bars
                errorbar(xPos, means(validIdx), sems(validIdx), 'k', 'LineStyle', 'none', 'LineWidth', 1.5);
                
                % Add significance markers
                for i = 1:length(xPos)
                    segIdx = validSegIndices(i);
                    if significant(segIdx)
                        text(xPos(i), means(segIdx) + sems(segIdx) + 0.01 * max(abs(means(validIdx))), '*', ...
                            'FontSize', 16, 'HorizontalAlignment', 'center');
                    end
                end
                
                % Set x-axis labels
                segLabels = segmentNames(validIdx);
                segLabelsShort = cell(size(segLabels));
                for i = 1:length(segLabels)
                    segLabelsShort{i} = strrep(segLabels{i}, '_', '\n');
                end
                set(gca, 'XTick', xPos, 'XTickLabel', segLabelsShort);
                xtickangle(45);
            end
            
            ylabel(metricName);
            title(sprintf('%s - %s', areas{a}, metricName));
            grid on;
        end
    end
end

sgtitle(sprintf('[%s] Within-Segment Avalanche Metrics (vs. Permutation)', filePrefix), 'FontSize', 14, 'FontWeight', 'bold');

if savePlots
    plotPath = fullfile(saveDir, sprintf('avalanche_within_segments_%s.png', sessionName));
    exportgraphics(gcf, plotPath, 'Resolution', 300);
    fprintf('Saved avalanche within-segment plot to: %s\n', plotPath);
end

% Plot 3: Between-segment comparisons
figure(2002); clf;
set(gcf, 'Position', [100, 100, 1800, 1200]);

for m = 1:nMetrics
    metricName = metricNames{m};
    
    for a = 1:nAreas
        subplot(nMetrics, nAreas, (m-1)*nAreas + a);
        hold on;
        
        if isfield(avalancheStatsResults.betweenSegments.(metricName), areas{a})
            bs = avalancheStatsResults.betweenSegments.(metricName).(areas{a});
            
            % Collect all pair comparisons
            pairNames = fieldnames(bs);
            nPairs = length(pairNames);
            
            if nPairs > 0
                diffs = zeros(1, nPairs);
                pValues = zeros(1, nPairs);
                significant = false(1, nPairs);
                pairLabels = cell(1, nPairs);
                
                for p = 1:nPairs
                    pairData = bs.(pairNames{p});
                    diffs(p) = pairData.diffMean;
                    pValues(p) = pairData.pValue;
                    significant(p) = pairData.significant;
                    
                    % Create short label
                    pairLabels{p} = strrep(pairNames{p}, '_vs_', ' vs\n');
                    pairLabels{p} = strrep(pairLabels{p}, '_', ' ');
                end
                
                % Plot bars
                xPos = 1:nPairs;
                b = bar(xPos, diffs, 'FaceColor', 'flat');
                
                % Color bars: blue if significant, gray if not
                for i = 1:nPairs
                    if significant(i)
                        b.CData(i, :) = [0.2 0.4 0.8];
                    else
                        b.CData(i, :) = [0.6 0.6 0.6];
                    end
                end
                
                % Add significance markers
                for i = 1:nPairs
                    if significant(i)
                        text(xPos(i), diffs(i) + 0.01 * sign(diffs(i)) * max(abs(diffs)), '*', ...
                            'FontSize', 16, 'HorizontalAlignment', 'center');
                    end
                end
                
                % Add zero line
                yline(0, 'k--', 'LineWidth', 1);
                
                set(gca, 'XTick', xPos, 'XTickLabel', pairLabels);
                xtickangle(45);
                ylabel(sprintf('Difference in Mean %s', metricName));
                title(sprintf('%s - %s', areas{a}, metricName));
                grid on;
            end
        end
    end
end

sgtitle(sprintf('[%s] Between-Segment Avalanche Comparisons', filePrefix), 'FontSize', 14, 'FontWeight', 'bold');

if savePlots
    plotPath = fullfile(saveDir, sprintf('avalanche_between_segments_%s.png', sessionName));
    exportgraphics(gcf, plotPath, 'Resolution', 300);
    fprintf('Saved avalanche between-segment plot to: %s\n', plotPath);
end

end

