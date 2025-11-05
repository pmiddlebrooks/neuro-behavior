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
periReachWindow = 30; % peri-reach window around each reach (used for all analyses)
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

% Helper to compute entropy from integer labels
computeEntropy = @(x) (isempty(x) || all(isnan(x))) * NaN + (~(isempty(x) || all(isnan(x)))) * ( ...
    -sum( max(eps, histcounts(x(~isnan(x)), 'BinMethod','integers','Normalization','probability')) ...
         .* log2(max(eps, histcounts(x(~isnan(x)), 'BinMethod','integers','Normalization','probability'))) ) );

for a = areasToTest
    if ~hasHmmArea(a), continue; end
    hmmRes = hmmResults{a};
    binSizeA = binSizes(a);
    winBinsA = ceil(periReachWindow / binSizeA);
    halfWinA = floor(winBinsA/2);
    nFramesPeri = winBinsA + 1;

    % local step/size
    stepBins = max(1, round((isempty(metricsStepSec) || isnan(metricsStepSec) || metricsStepSec<=0)*binSizeA/binSizeA + (~(isempty(metricsStepSec) || isnan(metricsStepSec) || metricsStepSec<=0))*metricsStepSec/binSizeA));
    metWinBins = max(1, round(slidingWindowSize / binSizeA));

    seq = hmmRes.continuous_results.sequence;
    hasMeta = isfield(hmmRes,'metastate_results') && isfield(hmmRes.metastate_results,'continuous_metastates') && ~isempty(hmmRes.metastate_results.continuous_metastates);
    if hasMeta, metaSeq = hmmRes.metastate_results.continuous_metastates; else, metaSeq = []; end
    tA = (0:length(seq)-1) * binSizeA;
    mapIdx = @(tSec) max(1, min(length(reachMaskTimeline), round(tSec/refBinSize)+1));

    stateEntropyWindows{a} = nan(length(reachStart), nFramesPeri);
    metaEntropyWindows{a} = nan(length(reachStart), nFramesPeri);
    behaviorSwitchesWindows{a} = nan(length(reachStart), nFramesPeri);
    behaviorProportionWindows{a} = nan(length(reachStart), nFramesPeri);
    behaviorEntropyWindows{a} = nan(length(reachStart), nFramesPeri);
    kinPCAStdWindows{a} = nan(length(reachStart), nFramesPeri);

    for r = 1:length(reachStart)
        centerTime = reachStart(r)/1000;
        [~, cIdx] = min(abs(tA - centerTime));
        periStart = cIdx - halfWinA; periEnd = cIdx + halfWinA;
        if periStart < 1 || periEnd > length(seq), continue; end

        for p = periStart:stepBins:periEnd
            swStart = max(periStart, p - floor(metWinBins/2));
            swEnd = min(periEnd, swStart + metWinBins - 1);
            swStart = max(periStart, swEnd - metWinBins + 1);
            if swEnd <= swStart, continue; end
            periPos = (p - periStart) + 1;

            % State entropy
            segS = seq(swStart:swEnd); segS = segS(segS>0);
            stateEntropyWindows{a}(r, periPos) = computeEntropy(segS);

            % Metastate entropy
            if hasMeta
                segM = metaSeq(swStart:swEnd); segM = segM(segM>0);
                metaEntropyWindows{a}(r, periPos) = computeEntropy(segM);
            end

            % Behavior metrics from reach/no-reach timeline
            tStart = (swStart-1)*binSizeA; tEnd = (swEnd-1)*binSizeA;
            bStart = mapIdx(tStart); bEnd = mapIdx(tEnd);
            if bEnd >= bStart
                segB = reachMaskTimeline(bStart:bEnd);
                if ~isempty(segB)
                    behaviorSwitchesWindows{a}(r, periPos) = sum(diff(segB)~=0);
                    behaviorProportionWindows{a}(r, periPos) = mean(segB);
                    p0 = mean(segB==0); p1 = mean(segB==1); pr = [p0 p1]; pr = pr(pr>0);
                    behaviorEntropyWindows{a}(r, periPos) = -sum(pr.*log2(pr));
                end
            end

            % Kin PCA std proxy from joystick amplitude
            if ~isempty(kinDown)
                kStart = bStart; kEnd = min(bEnd, length(kinDown));
                if kEnd >= kStart
                    kinPCAStdWindows{a}(r, periPos) = std(kinDown(kStart:kEnd));
                end
            end
        end
    end
end

% ==============================================     Plotting Results     ==============================================

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
filename = fullfile(saveDir, sprintf('%s_peri_reach_hmm_states_win%gs.png', filePrefix, periReachWindow));
exportgraphics(gcf, filename, 'Resolution', 300);
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
    filename = fullfile(saveDir, sprintf('%s_peri_reach_metastates_win%gs.png', filePrefix, periReachWindow));
    exportgraphics(gcf, filename, 'Resolution', 300);
    fprintf('Saved peri-reach metastate plot to: %s\n', filename);
else
    fprintf('\nNo metastate data available for plotting\n');
end

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
