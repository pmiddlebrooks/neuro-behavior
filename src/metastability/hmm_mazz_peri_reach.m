%%
% Peri-Reach HMM State Analysis
% Loads results from hmm_mazz_reach.m and analyzes HMM state sequences
% around reach onset times for each brain area
%
% Variables:
%   results - loaded HMM analysis results
%   dataR - reach behavioral data
%   areas - brain areas to analyze
%   reachStartFrame - reach start times in frame units for each area
%   stateWindows - HMM state sequences in windows around each reach
%   stateImages - imagesc plots of state sequences for each condition

%% Load existing results if requested
paths = get_paths;
binSize = .01;
minDur = .04;

% User-specified reach data file (should match the one used in hmm_mazz_reach.m)
% User-specified reach data file (should match the one used in criticality_reach_ar.m)
reachDataFile = fullfile(paths.reachDataPath, 'AB2_01-May-2023 15_34_59_NeuroBeh.mat');
% reachDataFile = fullfile(paths.reachDataPath, 'AB2_11-May-2023 17_31_00_NeuroBeh.mat');
% reachDataFile = fullfile(paths.reachDataPath, 'AB2_28-Apr-2023 17_50_02_NeuroBeh.mat');
% reachDataFile = fullfile(paths.reachDataPath, 'AB2_30-May-2023 12_49_52_NeuroBeh.mat');
% reachDataFile = fullfile(paths.reachDataPath, 'AB6_02-Apr-2025 14_18_54_NeuroBeh.mat');
% reachDataFile = fullfile(paths.reachDataPath, 'AB6_03-Apr-2025 13_34_09_NeuroBeh.mat');
% reachDataFile = fullfile(paths.reachDataPath, 'AB6_27-Mar-2025 14_04_12_NeuroBeh.mat');
% reachDataFile = fullfile(paths.reachDataPath, 'AB6_29-Mar-2025 15_21_05_NeuroBeh.mat');
reachDataFile = fullfile(paths.reachDataPath, 'Y4_06-Oct-2023 14_14_53_NeuroBeh.mat');
% reachDataFile = fullfile(paths.dropPath, 'reach_test.mat');

areasToTest = 1:4;

% Toggle plotting error reaches
plotErrors = true;

% Define window parameters
periReachWindow = 20; % peri-reach window around each reach (used for all analyses)
% Sliding metrics window (slides across the peri window for entropy/behavior/kin)
slidingWindowSize = 2.0;  % seconds
metricsStepSec = [];     % default: one HMM bin step (set per area)

% Extract session name from filename
[~, sessionName, ~] = fileparts(reachDataFile);

% Determine save directory based on loaded data file name (same as hmm_mazz_reach.m)
[~, dataBaseName, ~] = fileparts(reachDataFile);
saveDir = fullfile(paths.reachResultsPath, dataBaseName);
filename = sprintf('hmm_mazz_reach_bin%.3f_minDur%.3f.mat', binSize, minDur);
resultsPath = fullfile(saveDir, filename);

% Extract first 10 characters of filename for titles and file names
filePrefix = sessionName(1:min(10, length(sessionName)));

% Load HMM analysis results
fprintf('Loading HMM analysis results from: %s\n', resultsPath);
if ~exist(resultsPath, 'file')
    error('Results file not found: %s\nMake sure hmm_mazz_reach.m has been run for this dataset.', resultsPath);
end
results = load(resultsPath);
results = results.results;

% Extract areas and parameters
areas = results.areas;
binSizes = results.binSize;
numStates = results.numStates;
hmmResults = results.hmm_results;

% Prepare distinct colormaps per area for states and metastates
stateCmaps = cell(1, length(areas));
metaCmaps = cell(1, length(areas));
for a = 1:length(areas)
    try
        if numStates(a) > 0
            rgb = maxdistcolor(numStates(a), @sRGB_to_OKLab);
            stateCmaps{a} = [1 1 1; rgb]; % state 0 is white
        else
            stateCmaps{a} = lines(max(1, numStates(a)+1));
            stateCmaps{a}(1,:) = [1 1 1];
        end
    catch
        % Fallback if maxdistcolor not available
        ctmp = lines(max(1, numStates(a)+1));
        ctmp(1,:) = [1 1 1];
        stateCmaps{a} = ctmp;
    end
    % Metastate colormap (if available)
    if ~isempty(hmmResults{a}) && isfield(hmmResults{a}, 'metastate_results') && isfield(hmmResults{a}.metastate_results, 'num_metastates')
        nMeta = hmmResults{a}.metastate_results.num_metastates;
        try
            if nMeta > 0
                rgbm = maxdistcolor(nMeta, @sRGB_to_OKLab);
                metaCmaps{a} = [1 1 1; rgbm]; % metastate 0 is white
            else
                metaCmaps{a} = lines(1); metaCmaps{a}(1,:) = [1 1 1];
            end
        catch
            ctmp = lines(max(1, nMeta+1));
            ctmp(1,:) = [1 1 1];
            metaCmaps{a} = ctmp;
        end
    else
        metaCmaps{a} = [];
    end
end

% Load reach behavioral data
fprintf('Loading reach behavioral data from: %s\n', reachDataFile);
dataR = load(reachDataFile);

reachStart = dataR.R(:,1); % In seconds
reachStop = dataR.R(:,2);
reachAmp = dataR.R(:,3); % Amplitude of each reach (distance from 0)

% Use Block(:,3) for reach classification
reachClass = dataR.Block(:,3);

% Define reach conditions
reachStartCorr1 = reachStart(ismember(reachClass, 2)); % Block 1 correct
reachStartCorr2 = reachStart(ismember(reachClass, 4)); % Block 2 correct
reachStartErr1 = reachStart(ismember(reachClass, 1));  % Block 1 error
reachStartErr2 = reachStart(ismember(reachClass, 3));  % Block 2 error

% ==============================================     Peri-Reach Analysis     ==============================================

% Initialize storage for peri-reach state sequences (separate correct/error by block)
stateWindowsCorr1 = cell(1, length(areas));
stateWindowsCorr2 = cell(1, length(areas));
stateWindowsErr1 = cell(1, length(areas));
stateWindowsErr2 = cell(1, length(areas));

% Initialize storage for peri-reach metastate sequences (separate correct/error by block)
metastateWindowsCorr1 = cell(1, length(areas));
metastateWindowsCorr2 = cell(1, length(areas));
metastateWindowsErr1 = cell(1, length(areas));
metastateWindowsErr2 = cell(1, length(areas));

% Flags indicating availability of data to analyze/plot per area
hasHmmArea = false(1, length(areas));

fprintf('\n=== Peri-Reach HMM State Analysis ===\n');

for a = areasToTest
    fprintf('\nProcessing area %s...\n', areas{a});

    % Get HMM results for this area
    hmmRes = hmmResults{a};
    
    % Check if HMM analysis was successful for this area
    if isempty(hmmRes) || ~strcmp(hmmRes.metadata.analysis_status, 'SUCCESS')
        fprintf('Skipping area %s due to failed HMM analysis.\n', areas{a});
        continue
    end
    
    hasHmmArea(a) = true;
    
    % Get continuous state sequence and parameters
    continuousSequence = hmmRes.continuous_results.sequence;
    binSize = hmmRes.HmmParam.BinSize;
    totalTimeBins = length(continuousSequence);
    
    % Get continuous metastate sequence if available
    if isfield(hmmRes, 'metastate_results') && ~isempty(hmmRes.metastate_results.continuous_metastates)
        continuousMetastates = hmmRes.metastate_results.continuous_metastates;
        hasMetastates = true;
    else
        continuousMetastates = [];
        hasMetastates = false;
    end
    
    % Calculate window duration in bins
    windowDurationBins = ceil(periReachWindow / binSize);
    halfWindow = floor(windowDurationBins / 2);
    
    % Initialize arrays for this area
    numReachesCorr1 = length(reachStartCorr1);
    numReachesCorr2 = length(reachStartCorr2);
    numReachesErr1 = length(reachStartErr1);
    numReachesErr2 = length(reachStartErr2);
    
    % Initialize storage for all reach windows (by block)
    stateWindowsCorr1{a} = nan(numReachesCorr1, windowDurationBins + 1);
    stateWindowsCorr2{a} = nan(numReachesCorr2, windowDurationBins + 1);
    stateWindowsErr1{a} = nan(numReachesErr1, windowDurationBins + 1);
    stateWindowsErr2{a} = nan(numReachesErr2, windowDurationBins + 1);
    
    % Initialize storage for metastate windows (by block)
    if hasMetastates
        metastateWindowsCorr1{a} = nan(numReachesCorr1, windowDurationBins + 1);
        metastateWindowsCorr2{a} = nan(numReachesCorr2, windowDurationBins + 1);
        metastateWindowsErr1{a} = nan(numReachesErr1, windowDurationBins + 1);
        metastateWindowsErr2{a} = nan(numReachesErr2, windowDurationBins + 1);
    end
    
    % Extract around each correct reach (Block 1)
    validCorr1 = 0;
    for r = 1:numReachesCorr1
        reachTime = reachStartCorr1(r)/1000; % Convert to seconds
        reachBin = round(reachTime / binSize) + 1; % Convert to bin index (1-indexed)
        
        winStart = reachBin - halfWindow;
        winEnd = reachBin + halfWindow;
        
        if winStart >= 1 && winEnd <= totalTimeBins
            stateWindowsCorr1{a}(r, :) = continuousSequence(winStart:winEnd);
            if hasMetastates
                metastateWindowsCorr1{a}(r, :) = continuousMetastates(winStart:winEnd);
            end
            validCorr1 = validCorr1 + 1;
        else
            stateWindowsCorr1{a}(r, :) = nan(1, windowDurationBins + 1);
            if hasMetastates
                metastateWindowsCorr1{a}(r, :) = nan(1, windowDurationBins + 1);
            end
        end
    end
    
    % Extract around each correct reach (Block 2)
    validCorr2 = 0;
    for r = 1:numReachesCorr2
        reachTime = reachStartCorr2(r)/1000; % Convert to seconds
        reachBin = round(reachTime / binSize) + 1; % Convert to bin index (1-indexed)
        
        winStart = reachBin - halfWindow;
        winEnd = reachBin + halfWindow;
        
        if winStart >= 1 && winEnd <= totalTimeBins
            stateWindowsCorr2{a}(r, :) = continuousSequence(winStart:winEnd);
            if hasMetastates
                metastateWindowsCorr2{a}(r, :) = continuousMetastates(winStart:winEnd);
            end
            validCorr2 = validCorr2 + 1;
        else
            stateWindowsCorr2{a}(r, :) = nan(1, windowDurationBins + 1);
            if hasMetastates
                metastateWindowsCorr2{a}(r, :) = nan(1, windowDurationBins + 1);
            end
        end
    end
    
    % Extract around each error reach (Block 1)
    validErr1 = 0;
    for r = 1:numReachesErr1
        reachTime = reachStartErr1(r)/1000; % Convert to seconds
        reachBin = round(reachTime / binSize) + 1; % Convert to bin index (1-indexed)
        
        winStart = reachBin - halfWindow;
        winEnd = reachBin + halfWindow;
        
        if winStart >= 1 && winEnd <= totalTimeBins
            stateWindowsErr1{a}(r, :) = continuousSequence(winStart:winEnd);
            if hasMetastates
                metastateWindowsErr1{a}(r, :) = continuousMetastates(winStart:winEnd);
            end
            validErr1 = validErr1 + 1;
        else
            stateWindowsErr1{a}(r, :) = nan(1, windowDurationBins + 1);
            if hasMetastates
                metastateWindowsErr1{a}(r, :) = nan(1, windowDurationBins + 1);
            end
        end
    end
    
    % Extract around each error reach (Block 2)
    validErr2 = 0;
    for r = 1:numReachesErr2
        reachTime = reachStartErr2(r)/1000; % Convert to seconds
        reachBin = round(reachTime / binSize) + 1; % Convert to bin index (1-indexed)
        
        winStart = reachBin - halfWindow;
        winEnd = reachBin + halfWindow;
        
        if winStart >= 1 && winEnd <= totalTimeBins
            stateWindowsErr2{a}(r, :) = continuousSequence(winStart:winEnd);
            if hasMetastates
                metastateWindowsErr2{a}(r, :) = continuousMetastates(winStart:winEnd);
            end
            validErr2 = validErr2 + 1;
        else
            stateWindowsErr2{a}(r, :) = nan(1, windowDurationBins + 1);
            if hasMetastates
                metastateWindowsErr2{a}(r, :) = nan(1, windowDurationBins + 1);
            end
        end
    end
    
    fprintf('Area %s: Corr B1=%d, Corr B2=%d, Err B1=%d, Err B2=%d valid reaches\n', areas{a}, validCorr1, validCorr2, validErr1, validErr2);
end

% ==============================================     Sliding Window Metrics (State/Metastate/Behavior/Kin)     ==============================================

% Initialize storage for sliding-window metrics per area
stateEntropyWindows = cell(1, length(areas));        % (reaches x frames)
metaEntropyWindows = cell(1, length(areas));         % (reaches x frames)
behaviorSwitchesWindows = cell(1, length(areas));    % (reaches x frames)
behaviorProportionWindows = cell(1, length(areas));  % (reaches x frames)
behaviorEntropyWindows = cell(1, length(areas));     % (reaches x frames)
kinPCAStdWindows = cell(1, length(areas));           % (reaches x frames)

% Build a binary reach timeline at reference binning for behavior metrics (0=non-reach,1=reach)
refBinSize = binSizes(find(binSizes>0,1,'first')); if isempty(refBinSize), refBinSize = 0.01; end
recordingDurationSec = max([reachStart; reachStop]);
bhvTimelineLen = ceil(recordingDurationSec / refBinSize);
reachMaskTimeline = zeros(bhvTimelineLen,1);
for r = 1:length(reachStart)
    rs = max(1, floor(reachStart(r)/1000/refBinSize)+1);
    re = min(bhvTimelineLen, max(rs, ceil(reachStop(r)/1000/refBinSize)));
    reachMaskTimeline(rs:re) = 1;
end

% Optional kinematics (joystick amplitude) downsampled to refBinSize
hasKin = isfield(dataR,'NIDATA') && size(dataR.NIDATA,1) >= 8;
if hasKin
    jsAmp = sqrt(double(dataR.NIDATA(7,2:end)).^2 + double(dataR.NIDATA(8,2:end)).^2)';
    srcFs = 1000; % Hz
    binsPerFrame = max(1, round(refBinSize * srcFs));
    numFramesKin = floor(length(jsAmp)/binsPerFrame);
    kinDown = zeros(numFramesKin,1);
    for i=1:numFramesKin
        sIdx = (i-1)*binsPerFrame + 1; eIdx = i*binsPerFrame;
        kinDown(i) = mean(jsAmp(sIdx:eIdx), 'omitnan');
    end
    kinDown = zscore(kinDown);
else
    kinDown = [];
end

% Note: Entropy computation now matches hmm_correlations.m approach (inline code below)

for a = areasToTest
    if ~hasHmmArea(a), continue; end
    hmmRes = hmmResults{a};
    binSizeA = binSizes(a);
    winBinsA = ceil(periReachWindow / binSizeA);
    halfWinA = floor(winBinsA/2);
    nFramesPeri = winBinsA + 1;

    % local step/size
    if isempty(metricsStepSec) || isnan(metricsStepSec) || metricsStepSec <= 0
        stepBins = 1; % Default: one HMM bin step
    else
        stepBins = max(1, round(metricsStepSec / binSizeA));
    end
    metWinBins = max(1, round(slidingWindowSize / binSizeA));

    seq = hmmRes.continuous_results.sequence;
    
    % Check if sequence has any valid states
    if isempty(seq) || all(seq == 0) || all(isnan(seq))
        fprintf('Warning: Area %s has no valid states in sequence (all zeros or NaN)\n', areas{a});
        continue;
    end
    hasMeta = isfield(hmmRes,'metastate_results') && isfield(hmmRes.metastate_results,'continuous_metastates') && ~isempty(hmmRes.metastate_results.continuous_metastates);
    if hasMeta, metaSeq = hmmRes.metastate_results.continuous_metastates; else, metaSeq = []; end
    tA = (0:length(seq)-1) * binSizeA;
    mapIdx = @(tSec) max(1, min(length(reachMaskTimeline), round(tSec/refBinSize)+1));

    % ========== STEP 1: Calculate sliding window metrics across entire dataset ==========
    numBins = length(seq);
    stateEntropyFull = nan(numBins, 1);
    metaEntropyFull = nan(numBins, 1);
    behaviorSwitchesFull = nan(numBins, 1);
    behaviorProportionFull = nan(numBins, 1);
    behaviorEntropyFull = nan(numBins, 1);
    kinPCAStdFull = nan(numBins, 1);
    
    halfMetWin = floor(metWinBins / 2);
    
    fprintf('Area %s: Calculating sliding window metrics across %d bins...\n', areas{a}, numBins);
    for p = 1:numBins
        % Define sliding window boundaries centered at p
        % The metric at position p should represent a window centered at p
        % Window spans from p - halfMetWin to p + halfMetWin (or as close as possible)
        swStart = max(1, p - halfMetWin);
        swEnd = min(numBins, p + halfMetWin);
        
        % Ensure window has minimum size (metWinBins)
        actualWinSize = swEnd - swStart + 1;
        if actualWinSize < metWinBins
            if swStart == 1
                % At start: extend window to the right to maintain size
                swEnd = min(numBins, swStart + metWinBins - 1);
            elseif swEnd == numBins
                % At end: extend window to the left to maintain size
                swStart = max(1, swEnd - metWinBins + 1);
            end
        end
        
        if swEnd <= swStart, continue; end
        
        % The metric at position p represents the window centered at p
        % Time alignment: metric at bin p corresponds to time at bin p in original sequence
        
        % State entropy
        segS = seq(swStart:swEnd);
        segS = segS(segS > 0); % Remove zeros (unassigned states)
        if isempty(segS) || all(isnan(segS))
            stateEntropyFull(p) = NaN;
        elseif length(unique(segS)) <= 1
            stateEntropyFull(p) = 0;
        else
            stateCounts = histcounts(segS, 1:(numStates(a)+1));
            stateProportions = stateCounts / length(segS);
            stateProportions = stateProportions(stateProportions > 0);
            if ~isempty(stateProportions)
                stateEntropyFull(p) = -sum(stateProportions .* log2(stateProportions));
            end
        end
        
        % Metastate entropy
        if hasMeta
            segM = metaSeq(swStart:swEnd);
            segM = segM(segM > 0);
            if isempty(segM) || all(isnan(segM))
                metaEntropyFull(p) = NaN;
            elseif length(unique(segM)) <= 1
                metaEntropyFull(p) = 0;
            else
                maxMetastate = hmmRes.metastate_results.num_metastates;
                metaCounts = histcounts(segM, 1:(maxMetastate+1));
                metaProportions = metaCounts / length(segM);
                metaProportions = metaProportions(metaProportions > 0);
                if ~isempty(metaProportions)
                    metaEntropyFull(p) = -sum(metaProportions .* log2(metaProportions));
                end
            end
        end
        
        % Behavior metrics from reach/no-reach timeline
        tStart = (swStart-1) * binSizeA;
        tEnd = (swEnd-1) * binSizeA;
        bStart = mapIdx(tStart);
        bEnd = mapIdx(tEnd);
        if bEnd >= bStart
            segB = reachMaskTimeline(bStart:bEnd);
            if ~isempty(segB)
                behaviorSwitchesFull(p) = sum(diff(segB) ~= 0);
                behaviorProportionFull(p) = mean(segB);
                p0 = mean(segB==0); p1 = mean(segB==1); pr = [p0 p1]; pr = pr(pr>0);
                behaviorEntropyFull(p) = -sum(pr.*log2(pr));
            end
        end
        
        % Kin PCA std proxy from joystick amplitude
        if ~isempty(kinDown)
            kStart = bStart;
            kEnd = min(bEnd, length(kinDown));
            if kEnd >= kStart
                kinPCAStdFull(p) = std(kinDown(kStart:kEnd), 'omitnan');
            end
        end
    end
    
    % ========== STEP 2: Extract peri-event windows from pre-calculated metrics ==========
    stateEntropyWindows{a} = nan(length(reachStart), nFramesPeri);
    metaEntropyWindows{a} = nan(length(reachStart), nFramesPeri);
    behaviorSwitchesWindows{a} = nan(length(reachStart), nFramesPeri);
    behaviorProportionWindows{a} = nan(length(reachStart), nFramesPeri);
    behaviorEntropyWindows{a} = nan(length(reachStart), nFramesPeri);
    kinPCAStdWindows{a} = nan(length(reachStart), nFramesPeri);
    
    fprintf('  Extracting peri-event windows for %d reaches...\n', length(reachStart));
    for r = 1:length(reachStart)
        centerTime = reachStart(r)/1000;
        [~, cIdx] = min(abs(tA - centerTime));
        periStart = cIdx - halfWinA;
        periEnd = cIdx + halfWinA;
        
        if periStart < 1 || periEnd > numBins
            continue;
        end
        
        % Extract windows from pre-calculated metrics
        periIndices = periStart:stepBins:periEnd;
        periIndices = periIndices(periIndices >= 1 & periIndices <= numBins);
        
        for idx = 1:length(periIndices)
            p = periIndices(idx);
            periPos = (p - periStart) + 1;
            if periPos > 0 && periPos <= nFramesPeri
                stateEntropyWindows{a}(r, periPos) = stateEntropyFull(p);
                if hasMeta
                    metaEntropyWindows{a}(r, periPos) = metaEntropyFull(p);
                end
                behaviorSwitchesWindows{a}(r, periPos) = behaviorSwitchesFull(p);
                behaviorProportionWindows{a}(r, periPos) = behaviorProportionFull(p);
                behaviorEntropyWindows{a}(r, periPos) = behaviorEntropyFull(p);
                if ~isempty(kinDown)
                    kinPCAStdWindows{a}(r, periPos) = kinPCAStdFull(p);
                end
            end
        end
    end
    
    fprintf('  Completed area %s\n', areas{a});
end

%% ==============================================     Plotting Results     ==============================================

% Create peri-reach plots for each area: conditions x areas
figure(400); clf;
% Prefer plotting on second screen if available
monitorPositions = get(0, 'MonitorPositions');
monitorTwo = monitorPositions(size(monitorPositions, 1), :);
if size(monitorPositions, 1) >= 2
    set(gcf, 'Position', monitorTwo);
else
    set(gcf, 'Position', monitorPositions(1, :));
end

% Determine number of conditions to plot
% Order: Block 1 Correct, Block 1 Error, Block 2 Correct, Block 2 Error
numConditions = 2; % Block 1 Correct, Block 1 Error
if plotErrors
    numConditions = numConditions + 2; % Add Block 2 Correct, Block 2 Error
end

% Use tight_subplot for layout: conditions (rows) x areas (columns)
numCols = length(areasToTest);
ha = tight_subplot(numConditions, numCols, [0.08 0.04], [0.08 0.1], [0.06 0.04]);

% Create time axis for peri-reach window (centered on reach onset)
timeAxisPeriReach = (-halfWindow:halfWindow) * binSizes(1); % Use first area's bin size for time axis

% Define colors for different conditions
colors = {'k', [0 0 .6], [.6 0 0], [0 0.6 0]};

% Plot each condition (row) and area (column)
plotIdx = 0;
for condIdx = 1:numConditions
    for areaIdx = 1:length(areasToTest)
        a = areasToTest(areaIdx);
        plotIdx = plotIdx + 1;
        
        axes(ha(plotIdx));
        hold on;
        
        % Determine which condition to plot
        % Order: Block 1 Correct, Block 1 Error, Block 2 Correct, Block 2 Error
        if condIdx == 1
            % Block 1 Correct
            stateData = stateWindowsCorr1{a};
            condName = 'Block 1 Correct';
            colorIdx = 1;
        elseif condIdx == 2 && plotErrors
            % Block 1 Error
            stateData = stateWindowsErr1{a};
            condName = 'Block 1 Error';
            colorIdx = 2;
        elseif (condIdx == 2 && ~plotErrors) || (condIdx == 3 && plotErrors)
            % Block 2 Correct
            stateData = stateWindowsCorr2{a};
            condName = 'Block 2 Correct';
            colorIdx = 3;
        elseif condIdx == 4 && plotErrors
            % Block 2 Error
            stateData = stateWindowsErr2{a};
            condName = 'Block 2 Error';
            colorIdx = 4;
        else
            continue; % Skip if not plotting errors
        end
        
        % Check if we have data for this area and condition
        if ~hasHmmArea(a) || isempty(stateData) || all(isnan(stateData(:)))
            % No data - show blank plot
            xlim([timeAxisPeriReach(1), timeAxisPeriReach(end)]);
            ylim([0.5, 1.5]);
            title(sprintf('%s - %s\n(No Data)', areas{a}, condName), 'FontSize', 10);
            continue;
        end
        
        % Remove rows that are all NaN and ensure trials are in order
        validRows = ~all(isnan(stateData), 2);
        if ~any(validRows)
            xlim([timeAxisPeriReach(1), timeAxisPeriReach(end)]);
            ylim([0.5, 1.5]);
            title(sprintf('%s - %s\n(No Valid Reaches)', areas{a}, condName), 'FontSize', 10);
            continue;
        end
        
        % Keep trials in original order (top to bottom)
        stateDataValid = stateData(validRows, :);
        
        % Create imagesc plot
        imagesc(timeAxisPeriReach, 1:size(stateDataValid, 1), stateDataValid);
        
        % Set colormap to be consistent within area; state 0 is white
        if ~isempty(stateCmaps{a})
            colormap(ha(plotIdx), stateCmaps{a});
            caxis(ha(plotIdx), [0, max(1, numStates(a))]);
        end
        
        % Add colorbar only on top row, and keep fixed ticks/labels per area
        if condIdx == 1
            c = colorbar('peer', ha(plotIdx));
            c.Ticks = 0:max(1, numStates(a));
            c.TickLabels = string(0:max(1, numStates(a)));
            c.Label.String = 'HMM State';
            c.Label.FontSize = 8;
        end
        
        % Add vertical line at reach onset
        plot([0 0], ylim, 'k--', 'LineWidth', 2);
        
        % Formatting
        xlabel('Time relative to reach onset (s)', 'FontSize', 8);
        ylabel('Reach Trial', 'FontSize', 8);
        title(sprintf('%s - %s\n(%d trials)', areas{a}, condName, sum(validRows)), 'FontSize', 10);
        
        % Set axis limits
        xlim([timeAxisPeriReach(1), timeAxisPeriReach(end)]);
        ylim([0.5, size(stateDataValid, 1) + 0.5]);
        
        % Set tick labels
        xTicks = ceil(timeAxisPeriReach(1)):floor(timeAxisPeriReach(end));
        if isempty(xTicks)
            xTicks = linspace(timeAxisPeriReach(1), timeAxisPeriReach(end), 5);
        end
        xticks(xTicks);
        xticklabels(string(xTicks));
        
        % Set y-axis ticks to show trial numbers
        if size(stateDataValid, 1) <= 10
            yticks(1:size(stateDataValid, 1));
        else
            yticks(1:5:size(stateDataValid, 1));
        end
        
        grid on;
        set(gca, 'GridAlpha', 0.3);
    end
end

% Add overall title
sgtitle(sprintf('%s - Peri-Reach HMM State Sequences (Window: %gs)', filePrefix, periReachWindow), 'FontSize', 16);

% Save combined figure (in same data-specific folder)
filename = fullfile(saveDir, sprintf('%s_peri_reach_hmm_states_win%gs.eps', filePrefix, periReachWindow));
exportgraphics(gcf, filename, 'ContentType', 'vector');
fprintf('Saved peri-reach HMM state plot to: %s\n', filename);

% ==============================================     Metastate Plotting     ==============================================

% Check if any area has metastate data
hasAnyMetastates = false;
for a = areasToTest
    if hasHmmArea(a) && ~isempty(results.hmm_results{a}) && isfield(results.hmm_results{a}, 'metastate_results') && ~isempty(results.hmm_results{a}.metastate_results.continuous_metastates)
        hasAnyMetastates = true;
        break;
    end
end

if hasAnyMetastates
    fprintf('\n=== Creating Peri-Reach Metastate Plot ===\n');
    
    % Create peri-reach metastate plots for each area: conditions x areas
    figure(401); clf;
    % Prefer plotting on second screen if available
    monitorPositions = get(0, 'MonitorPositions');
    monitorTwo = monitorPositions(size(monitorPositions, 1), :);
    if size(monitorPositions, 1) >= 2
        set(gcf, 'Position', monitorTwo);
    else
        set(gcf, 'Position', monitorPositions(1, :));
    end

    % Use tight_subplot for layout: conditions (rows) x areas (columns)
    numCols = length(areasToTest);
    ha = tight_subplot(numConditions, numCols, [0.08 0.04], [0.08 0.1], [0.06 0.04]);

    % Plot each condition (row) and area (column)
    plotIdx = 0;
    for condIdx = 1:numConditions
        for areaIdx = 1:length(areasToTest)
            a = areasToTest(areaIdx);
            plotIdx = plotIdx + 1;
            
            axes(ha(plotIdx));
            hold on;
            
            % Determine which condition to plot
            % Order: Block 1 Correct, Block 1 Error, Block 2 Correct, Block 2 Error
            if condIdx == 1
                % Block 1 Correct
                metastateData = metastateWindowsCorr1{a};
                condName = 'Block 1 Correct';
                colorIdx = 1;
            elseif condIdx == 2 && plotErrors
                % Block 1 Error
                metastateData = metastateWindowsErr1{a};
                condName = 'Block 1 Error';
                colorIdx = 2;
            elseif (condIdx == 2 && ~plotErrors) || (condIdx == 3 && plotErrors)
                % Block 2 Correct
                metastateData = metastateWindowsCorr2{a};
                condName = 'Block 2 Correct';
                colorIdx = 3;
            elseif condIdx == 4 && plotErrors
                % Block 2 Error
                metastateData = metastateWindowsErr2{a};
                condName = 'Block 2 Error';
                colorIdx = 4;
            else
                continue; % Skip if not plotting errors
            end
            
            % Check if we have metastate data for this area and condition
            if ~hasHmmArea(a) || isempty(metastateData) || all(isnan(metastateData(:)))
                % No data - show blank plot
                xlim([timeAxisPeriReach(1), timeAxisPeriReach(end)]);
                ylim([0.5, 1.5]);
                title(sprintf('%s - %s\n(No Metastate Data)', areas{a}, condName), 'FontSize', 10);
                continue;
            end
            
            % Remove rows that are all NaN and ensure trials are in order
            validRows = ~all(isnan(metastateData), 2);
            if ~any(validRows)
                xlim([timeAxisPeriReach(1), timeAxisPeriReach(end)]);
                ylim([0.5, 1.5]);
                title(sprintf('%s - %s\n(No Valid Reaches)', areas{a}, condName), 'FontSize', 10);
                continue;
            end
            
            % Keep trials in original order (top to bottom)
            metastateDataValid = metastateData(validRows, :);
            
            % Create imagesc plot
            imagesc(timeAxisPeriReach, 1:size(metastateDataValid, 1), metastateDataValid);
            
            % Set colormap for metastates; metastate 0 is white
            if ~isempty(metaCmaps{a})
                colormap(ha(plotIdx), metaCmaps{a});
                nMeta = size(metaCmaps{a},1) - 1;
                caxis(ha(plotIdx), [0, max(1, nMeta)]);
            end
            
            % Add colorbar only on top row, and keep fixed ticks/labels per area
            if condIdx == 1
                c = colorbar('peer', ha(plotIdx));
                if exist('nMeta','var') && ~isempty(nMeta)
                    c.Ticks = 0:max(1, nMeta);
                    c.TickLabels = string(0:max(1, nMeta));
                end
                c.Label.String = 'Metastate';
                c.Label.FontSize = 8;
            end
            
            % Add vertical line at reach onset
            plot([0 0], ylim, 'k--', 'LineWidth', 2);
            
            % Formatting
            xlabel('Time relative to reach onset (s)', 'FontSize', 8);
            ylabel('Reach Trial', 'FontSize', 8);
            title(sprintf('%s - %s\n(%d trials)', areas{a}, condName, sum(validRows)), 'FontSize', 10);
            
            % Set axis limits
            xlim([timeAxisPeriReach(1), timeAxisPeriReach(end)]);
            ylim([0.5, size(metastateDataValid, 1) + 0.5]);
            
            % Set tick labels
            xTicks = ceil(timeAxisPeriReach(1)):floor(timeAxisPeriReach(end));
            if isempty(xTicks)
                xTicks = linspace(timeAxisPeriReach(1), timeAxisPeriReach(end), 5);
            end
            xticks(xTicks);
            xticklabels(string(xTicks));
            
            % Set y-axis ticks to show trial numbers
            if size(metastateDataValid, 1) <= 10
                yticks(1:size(metastateDataValid, 1));
            else
                yticks(1:5:size(metastateDataValid, 1));
            end
            
            grid on;
            set(gca, 'GridAlpha', 0.3);
        end
    end

    % Add overall title
    sgtitle(sprintf('%s - Peri-Reach Metastate Sequences (Window: %gs)', filePrefix, periReachWindow), 'FontSize', 16);

    % Save combined metastate figure (in same data-specific folder)
    filename = fullfile(saveDir, sprintf('%s_peri_reach_metastates_win%gs.eps', filePrefix, periReachWindow));
    exportgraphics(gcf, filename, 'ContentType', 'vector');
    fprintf('Saved peri-reach metastate plot to: %s\n', filename);
else
    fprintf('\nNo metastate data available for plotting\n');
end
%% ==============================================     Sliding Window Metrics Plotting     ==============================================

% Create condition indices (like criticality_peri_reach.m)
corr1Idx = ismember(reachClass, 2); % Block 1 correct
corr2Idx = ismember(reachClass, 4); % Block 2 correct
err1Idx = ismember(reachClass, 1);  % Block 1 error
err2Idx = ismember(reachClass, 3);  % Block 2 error

% Determine number of conditions to plot
numConditions = 2; % Block 1 Correct, Block 1 Error
if plotErrors
    numConditions = numConditions + 2; % Add Block 2 Correct, Block 2 Error
end

% Metrics to plot
metricsToPlot = {'stateEntropy', 'behaviorSwitches', 'behaviorEntropy', 'kinPCAStd'};
metricNames = {'State Entropy', 'Behavior Switches', 'Behavior Entropy', 'Kin PCA Std'};
metricWindows = {stateEntropyWindows, behaviorSwitchesWindows, behaviorEntropyWindows, kinPCAStdWindows};

% Create figure for metrics
figure(402); clf;
monitorPositions = get(0, 'MonitorPositions');
monitorTwo = monitorPositions(size(monitorPositions, 1), :);
if size(monitorPositions, 1) >= 2
    set(gcf, 'Position', monitorTwo);
else
    set(gcf, 'Position', monitorPositions(1, :));
end

% Layout: rows = conditions (4 if plotErrors, else 2), columns = areas
numCols = length(areasToTest);
ha_metrics = tight_subplot(numConditions, numCols, [0.08 0.04], [0.08 0.1], [0.06 0.04]);

% Create time axes for each area (same as peri-reach window)
timeAxisPerArea = cell(1, length(areas));
for a = areasToTest
    if hasHmmArea(a)
        binSizeA = binSizes(a);
        winBinsA = ceil(periReachWindow / binSizeA);
        halfWinA = floor(winBinsA / 2);
        timeAxisPerArea{a} = (-halfWinA:halfWinA) * binSizeA;
    end
end

% Colors for metrics (different colors for each metric)
metricColors = {[0 0.4470 0.7410], [0.8500 0.3250 0.0980], [0.9290 0.6940 0.1250], [0.4940 0.1840 0.5560]}; % Blue, Red-orange, Yellow, Purple
metricLineStyles = {'-', '--', ':', '-.'};

plotIdx = 0;
for condIdx = 1:numConditions
    % Determine which condition
    if condIdx == 1
        condIdx_use = corr1Idx;
        condName = 'Block 1 Correct';
    elseif condIdx == 2 && plotErrors
        condIdx_use = err1Idx;
        condName = 'Block 1 Error';
    elseif (condIdx == 2 && ~plotErrors) || (condIdx == 3 && plotErrors)
        condIdx_use = corr2Idx;
        condName = 'Block 2 Correct';
    elseif condIdx == 4 && plotErrors
        condIdx_use = err2Idx;
        condName = 'Block 2 Error';
    else
        continue;
    end
    
    % Plot each area
    for areaIdx = 1:length(areasToTest)
        a = areasToTest(areaIdx);
        plotIdx = plotIdx + 1;
        
        axes(ha_metrics(plotIdx));
        hold on;
        
        % Check if we have any data for this area
        hasData = false;
        for mIdx = 1:length(metricsToPlot)
            metricWindows_m = metricWindows{mIdx};
            if hasHmmArea(a) && ~isempty(metricWindows_m{a}) && any(~isnan(metricWindows_m{a}(:)))
                hasData = true;
                break;
            end
        end
        
        if ~hasData
            xlim([-periReachWindow/2, periReachWindow/2]);
            ylim([0 1]);
            title(sprintf('%s - %s\n(No Data)', areas{a}, condName), 'FontSize', 10);
            continue;
        end
        
        % Get time axis for this area
        if ~isempty(timeAxisPerArea{a})
            timeAxis = timeAxisPerArea{a};
        else
            % Fallback
            binSizeA = binSizes(a);
            winBinsA = ceil(periReachWindow / binSizeA);
            halfWinA = floor(winBinsA / 2);
            timeAxis = (-halfWinA:halfWinA) * binSizeA;
        end
        
        timeAxis_plot = timeAxis; % Initialize with full time axis
        
        % Plot each metric on the same axes
        for mIdx = 1:length(metricsToPlot)
            metricWindows_m = metricWindows{mIdx};
            
            % Check if data exists
            if ~hasHmmArea(a) || isempty(metricWindows_m{a}) || all(isnan(metricWindows_m{a}(:)))
                continue;
            end
            
            % Extract data for this condition
            metricData = metricWindows_m{a}(condIdx_use, :);
            validRows = ~all(isnan(metricData), 2);
            
            if ~any(validRows)
                continue;
            end
            
            metricDataValid = metricData(validRows, :);
            
            % Calculate mean and std
            meanMetric = nanmean(metricDataValid, 1);
            stdMetric = nanstd(metricDataValid, 0, 1);
            
            % Ensure timeAxis matches data length
            if length(timeAxis) ~= length(meanMetric)
                minLen = min(length(timeAxis), length(meanMetric));
                timeAxis_plot_metric = timeAxis(1:minLen);
                meanMetric = meanMetric(1:minLen);
                stdMetric = stdMetric(1:minLen);
            else
                timeAxis_plot_metric = timeAxis;
            end
            
            % Update timeAxis_plot to the shortest one (in case metrics have different lengths)
            if length(timeAxis_plot_metric) < length(timeAxis_plot)
                timeAxis_plot = timeAxis_plot_metric;
            end
            
            % Plot std ribbon
            fill([timeAxis_plot_metric, fliplr(timeAxis_plot_metric)], ...
                 [meanMetric + stdMetric, fliplr(meanMetric - stdMetric)], ...
                 metricColors{mIdx}, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', [metricNames{mIdx} ' ± std']);
            
            % Plot mean line
            plot(timeAxis_plot_metric, meanMetric, 'Color', metricColors{mIdx}, 'LineWidth', 2, ...
                 'LineStyle', metricLineStyles{mIdx}, 'DisplayName', metricNames{mIdx});
        end
        
        % Add vertical line at reach onset
        plot([0 0], ylim, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
        
        % Formatting
        xlabel('Time (s)', 'FontSize', 10);
        ylabel('Metric Value', 'FontSize', 10);
        title(sprintf('%s - %s', areas{a}, condName), 'FontSize', 10);
        grid on;
        
        % Set axis limits
        if ~isempty(timeAxis_plot)
            xlim([timeAxis_plot(1), timeAxis_plot(end)]);
        else
            xlim([-periReachWindow/2, periReachWindow/2]);
        end
        
        % Auto-scale y-axis with some padding
        ylim('auto');
        yLim = ylim;
        if ~isnan(yLim(1)) && ~isnan(yLim(2)) && yLim(2) > yLim(1)
            ylim([yLim(1) - 0.05*range(yLim), yLim(2) + 0.05*range(yLim)]);
        end
        
        % Set tick labels
        if ~isempty(timeAxis_plot)
            xTicks = ceil(timeAxis_plot(1)):floor(timeAxis_plot(end));
            if isempty(xTicks)
                xTicks = linspace(timeAxis_plot(1), timeAxis_plot(end), 5);
            end
            xticks(xTicks);
            xticklabels(string(xTicks));
        end
        
        % Add legend only on first subplot (top-left)
        if condIdx == 1 && areaIdx == 1
            legend('Location', 'best', 'FontSize', 8);
        end
    end
end

% Add overall title
sgtitle(sprintf('%s - Sliding Window Metrics (Window: %.1fs, Peri: %.1fs)', filePrefix, slidingWindowSize, periReachWindow), 'FontSize', 14);

% Save figure
filename_metrics = fullfile(saveDir, sprintf('%s_peri_reach_metrics_win%.1f_peri%.1f.eps', filePrefix, slidingWindowSize, periReachWindow));
exportgraphics(gcf, filename_metrics, 'ContentType', 'vector');
fprintf('Saved peri-reach metrics plot to: %s\n', filename_metrics);


%% ==============================================     Summary Statistics     ==============================================

fprintf('\n=== Peri-Reach HMM State Analysis Summary ===\n');
for a = areasToTest
    if hasHmmArea(a)
        fprintf('\nArea %s:\n', areas{a});
        fprintf('  Total reaches analyzed: %d\n', length(reachStart));
        fprintf('  Valid reaches Block 1 Correct: %d\n', sum(~all(isnan(stateWindowsCorr1{a}), 2)));
        fprintf('  Valid reaches Block 2 Correct: %d\n', sum(~all(isnan(stateWindowsCorr2{a}), 2)));
        if plotErrors
            fprintf('  Valid reaches Block 1 Error: %d\n', sum(~all(isnan(stateWindowsErr1{a}), 2)));
            fprintf('  Valid reaches Block 2 Error: %d\n', sum(~all(isnan(stateWindowsErr2{a}), 2)));
        end
        fprintf('  Number of HMM states: %d\n', numStates(a));
        fprintf('  Bin size: %.6f seconds\n', binSizes(a));
    else
        fprintf('\nArea %s: HMM analysis failed\n', areas{a});
    end
end

fprintf('\nPeri-reach HMM state analysis complete!\n');
