%%
% Peri-Reach Criticality Analysis
% Loads results from criticality_compare.m and analyzes d2 criticality values
% around reach onset times for each brain area
%
% Variables:
%   results - loaded criticality analysis results
%   dataR - reach behavioral data
%   areas - brain areas to analyze
%   reachStartFrame - reach start times in frame units for each area
%   d2Windows - d2 criticality values in 3-second windows around each reach
%   meanD2PeriReach - mean d2 values across all reaches for each area

%% Load existing results if requested
slidingWindowSize = 8;% For d2, use a small window to try to optimize temporal resolution

% User-specified reach data file (should match the one used in criticality_reach_ar.m)
reachDataFile = fullfile(paths.reachDataPath, 'AB2_01-May-2023 15_34_59_NeuroBeh.mat');
% reachDataFile = fullfile(paths.reachDataPath, 'AB2_11-May-2023 17_31_00_NeuroBeh.mat');
% reachDataFile = fullfile(paths.reachDataPath, 'AB2_28-Apr-2023 17_50_02_NeuroBeh.mat');
% reachDataFile = fullfile(paths.reachDataPath, 'AB2_30-May-2023 12_49_52_NeuroBeh.mat');
% reachDataFile = fullfile(paths.reachDataPath, 'AB6_02-Apr-2025 14_18_54_NeuroBeh.mat');
% reachDataFile = fullfile(paths.reachDataPath, 'AB6_03-Apr-2025 13_34_09_NeuroBeh.mat');
% reachDataFile = fullfile(paths.reachDataPath, 'AB6_27-Mar-2025 14_04_12_NeuroBeh.mat');
% reachDataFile = fullfile(paths.reachDataPath, 'AB6_29-Mar-2025 15_21_05_NeuroBeh.mat');
% reachDataFile = fullfile(paths.reachDataPath, 'Y4_06-Oct-2023 14_14_53_NeuroBeh.mat');
reachDataFile = fullfile(paths.reachDataPath, 'makeSpikes.mat');
% reachDataFile = fullfile(paths.dropPath, 'reach_task/data/Y4_06-Oct-2023 14_14_53_NeuroBeh.mat');

areasToTest = 1:4;

% Toggle plotting error reaches (single traces and mean)
plotErrors = true;
% Plotting toggles
plotD2 = true;     % plot d2 panels
plotMrBr = true;   % plot mrBr panels (only if d2 exists for area)

% NEW: Sliding window d2 calculation options
calculateSlidingWindowD2 = true;  % Set to true to perform sliding window d2 on mean-centered popActivity
d2Window = 60;  % Window duration in seconds for collecting popActivity per reach

%
paths = get_paths;

% Determine save directory based on loaded data file name (same as criticality_reach_ar.m)
[~, dataBaseName, ~] = fileparts(reachDataFile);
saveDir = fullfile(paths.reachResultsPath, dataBaseName);
resultsPathWin = fullfile(saveDir, sprintf('criticality_reach_ar_win%d.mat', slidingWindowSize));

% Extract first 10 characters of filename for titles and file names
filePrefix = dataBaseName(1:min(10, length(dataBaseName)));

% Load criticality analysis results
fprintf('Loading criticality analysis results from: %s\n', resultsPathWin);
if ~exist(resultsPathWin, 'file')
    error('Results file not found: %s\nMake sure criticality_reach_ar.m has been run for this dataset and window size.', resultsPathWin);
end
results = load(resultsPathWin);
results = results.results;

% Extract areas and parameters
areas = results.areas;
optimalBinSize = results.reach.optimalBinSize;
d2Rea = results.reach.d2;
mrBrRea = results.reach.mrBr;
startS = results.reach.startS;

% Load reach behavioral data
fprintf('Loading reach behavioral data from: %s\n', reachDataFile);
dataR = load(reachDataFile);

reachStart = dataR.R(:,1); % In seconds
% reachStop = dataR.R(:,2);
% reachAmp = dataR.R(:,3); % Amplitude of each reach (distance from 0)

 % block or BlockWithErrors - columns are BlockNum Correct? ReachClassification1-4 ReachClassification-20-20
 %     -rows are reaches
 %     classification1-4:
 %     - 1 - block 1 error
 %     - 2 - block 1 Rew
 %     - 3 - block 2 error
 %     - 4 - block 2 rew
 %     - 5 - block 3 error
 %     - 6 - block 3 rew
 % 
 % 
 %     classification-20-20:
 %     - -10 - block 1 error below
 %     -  1   - block 1 Rew
 %     - 10 - block 1 error above
 %     - -20 - block 2 error below
 %     -  2   - block 2 Rew
 %     - 20 - block 2 error above

    reachClass = dataR.Block(:,3);
% Calculate condition indices (same for all areas)
corr1Idx = ismember(reachClass, 2);
corr2Idx = ismember(reachClass, 4);
err1Idx = ismember(reachClass, 1);
err2Idx = ismember(reachClass, 3);

% ==============================================     Peri-Reach Analysis     ==============================================

% Define window parameters
windowDurationSec = 40; % 3-second window around each reach
windowDurationFrames = cell(1, length(areas));
for a = areasToTest
    windowDurationFrames{a} = ceil(windowDurationSec / optimalBinSize(a));
end

% Initialize storage for peri-reach d2/mrBr values (combined approach)
d2Windows = cell(1, length(areas));
mrBrWindows = cell(1, length(areas));
timeAxisPeriReach = cell(1, length(areas));

% Flags indicating availability of data to analyze/plot per area
hasD2Area = false(1, length(areas));
hasMrBrArea = false(1, length(areas));

% ==============================================     Prepare Shared Reach Data     ==============================================
fprintf('\n=== Preparing Shared Reach Data ===\n');

totalReaches = length(reachStart);
fprintf('Total reaches: %d (Corr1=%d, Corr2=%d, Err1=%d, Err2=%d)\n', totalReaches, ...
    sum(corr1Idx), sum(corr2Idx), sum(err1Idx), sum(err2Idx));

fprintf('\n=== Peri-Reach d2 Criticality Analysis ===\n');

for a = areasToTest
    fprintf('\nProcessing area %s...\n', areas{a});

    % Get d2/mrBr values and time points for this area
    d2Values = d2Rea{a};
    mrValues = mrBrRea{a};
    timePoints = startS{a};

    % Availability flags for this area (plotting will be based only on d2)
    hasD2Area(a) = ~isempty(d2Values) && any(~isnan(d2Values));
    hasMrBrArea(a) = ~isempty(mrValues) && any(~isnan(mrValues));

    % If no d2 for this area, skip analyses entirely for this area
    if ~hasD2Area(a)
        fprintf('Skipping area %s due to missing d2 data.\n', areas{a});
        continue
    end

    halfWindow = floor(windowDurationFrames{a} / 2);

    % Create time axis for peri-reach window (centered on reach onset)
    timeAxisPeriReach{a} = (-halfWindow:halfWindow) * optimalBinSize(a);

    % Initialize storage for all reach windows (combined)
    d2Windows{a} = nan(totalReaches, windowDurationFrames{a} + 1);
    mrBrWindows{a} = nan(totalReaches, windowDurationFrames{a} + 1);

    % Extract around each reach (combined approach)
    validReaches = 0;
    for r = 1:totalReaches
        reachTime = reachStart(r)/1000; % Convert to seconds
        [~, closestIdx] = min(abs(timePoints - reachTime));
        winStart = closestIdx - halfWindow;
        winEnd = closestIdx + halfWindow;
        
        if winStart >= 1 && winEnd <= length(timePoints)
            if hasD2Area(a)
                d2Windows{a}(r, :) = d2Values(winStart:winEnd);
            end
            if hasMrBrArea(a)
                mrBrWindows{a}(r, :) = mrValues(winStart:winEnd);
            end
            validReaches = validReaches + 1;
        else
            d2Windows{a}(r, :) = nan(1, windowDurationFrames{a} + 1);
            mrBrWindows{a}(r, :) = nan(1, windowDurationFrames{a} + 1);
        end
    end

    % Calculate mean d2/mrBr values by condition
    % Use shared condition indices
    if hasD2Area(a)
        meanD2PeriReachCorr1{a} = nanmean(d2Windows{a}(corr1Idx, :), 1);
        meanD2PeriReachCorr2{a} = nanmean(d2Windows{a}(corr2Idx, :), 1);
        meanD2PeriReachErr1{a} = nanmean(d2Windows{a}(err1Idx, :), 1);
        meanD2PeriReachErr2{a} = nanmean(d2Windows{a}(err2Idx, :), 1);
    end
    
    if hasMrBrArea(a)
        meanMrBrPeriReachCorr1{a} = nanmean(mrBrWindows{a}(corr1Idx, :), 1);
        meanMrBrPeriReachCorr2{a} = nanmean(mrBrWindows{a}(corr2Idx, :), 1);
        meanMrBrPeriReachErr1{a} = nanmean(mrBrWindows{a}(err1Idx, :), 1);
        meanMrBrPeriReachErr2{a} = nanmean(mrBrWindows{a}(err2Idx, :), 1);
    end

    fprintf('Area %s: Total=%d valid reaches (Corr1=%d, Corr2=%d, Err1=%d, Err2=%d)\n', areas{a}, validReaches, ...
        sum(corr1Idx), sum(corr2Idx), sum(err1Idx), sum(err2Idx));
end

% ==============================================     Sliding Window D2 Analysis     ==============================================

if calculateSlidingWindowD2
    fprintf('\n=== Sliding Window D2 Analysis on Mean-Centered Population Activity ===\n');
    
    % Get parameters from results
    pOrder = results.params.pOrder;
    critType = results.params.critType;
    
    % Initialize storage for sliding window d2 results (combined approach)
    slidingD2Windows = cell(1, length(areas));
    
    for a = areasToTest
        fprintf('\nProcessing sliding window d2 for area %s...\n', areas{a});
        
        % Get population activity for this area
        popActivityFull = results.popActivity{a};
        if isempty(popActivityFull)
            fprintf('Skipping area %s due to missing population activity data.\n', areas{a});
            continue;
        end
        
        % Get time points and bin size for this area
        timePoints = startS{a};
        binSize = optimalBinSize(a);
        
        % Calculate window parameters
        d2WindowFrames = ceil(d2Window / binSize);
        halfD2Window = floor(d2WindowFrames / 2);
        
        % Use shared reach data (same for all areas)
        % Initialize storage for all reach windows (combined)
        slidingD2Windows{a} = nan(totalReaches, windowDurationFrames{a} + 1);
        
        % Collect all popActivity windows for mean-centering
        allPopActivityWindows = [];
        
        % Collect popActivity windows for all reaches
        for r = 1:totalReaches
            reachTime = reachStart(r)/1000; % Convert to seconds
            [~, closestIdx] = min(abs(timePoints - reachTime));
            winStart = closestIdx - halfD2Window;
            winEnd = closestIdx + halfD2Window;
            
            if winStart >= 1 && winEnd <= length(popActivityFull)
                popWindow = popActivityFull(winStart:winEnd);
                allPopActivityWindows = [allPopActivityWindows; popWindow'];
            end
        end
        
        if isempty(allPopActivityWindows)
            fprintf('No valid popActivity windows found for area %s\n', areas{a});
            continue;
        end
        
        % Calculate mean popActivity across all reaches
        meanPopActivity = mean(allPopActivityWindows, 1);
        
        % Mean-center each popActivity window
        meanCenteredPopActivity = allPopActivityWindows - meanPopActivity;
        
        % Now perform sliding window d2 analysis on each mean-centered reach
        for r = 1:totalReaches
            reachTime = reachStart(r)/1000;
            [~, closestIdx] = min(abs(timePoints - reachTime));
            winStart = closestIdx - halfD2Window;
            winEnd = closestIdx + halfD2Window;
            
            if winStart >= 1 && winEnd <= length(popActivityFull)
                meanCenteredPop = meanCenteredPopActivity(r, :);
                
                % Perform sliding window d2 analysis
                slidingD2Values = nan(1, windowDurationFrames{a} + 1);
                
                % Calculate sliding window parameters
                stepSamples = round(results.d2StepSize(a) / binSize);
                winSamples = round(results.d2WindowSize(a) / binSize);
                
                % Ensure we have enough data
                if length(meanCenteredPop) >= winSamples
                    numSlidingWindows = floor((length(meanCenteredPop) - winSamples) / stepSamples) + 1;
                    
                    for w = 1:min(numSlidingWindows, windowDurationFrames{a} + 1)
                        startIdx = (w - 1) * stepSamples + 1;
                        endIdx = min(startIdx + winSamples - 1, length(meanCenteredPop));
                        
                        if endIdx - startIdx + 1 >= winSamples * 0.8 % Use at least 80% of window
                            wPopActivity = meanCenteredPop(startIdx:endIdx);
                            
                            % Calculate d2 using Yule-Walker
                            try
                                [varphi, ~] = myYuleWalker3(wPopActivity, pOrder);
                                slidingD2Values(w) = getFixedPointDistance2(pOrder, critType, varphi);
                            catch
                                slidingD2Values(w) = nan;
                            end
                        end
                    end
                end
                
                % Store in the appropriate window position
                winStartPeri = closestIdx - halfWindow;
                winEndPeri = closestIdx + halfWindow;
                if winStartPeri >= 1 && winEndPeri <= length(timePoints)
                    periStartIdx = max(1, winStartPeri - (closestIdx - halfD2Window) + 1);
                    periEndIdx = min(length(slidingD2Values), winEndPeri - (closestIdx - halfD2Window) + 1);
                    slidingD2Windows{a}(r, periStartIdx:periEndIdx) = slidingD2Values(periStartIdx:periEndIdx);
                end
            end
        end
        
        % Calculate mean sliding d2 values by condition
        % Use shared condition indices
        meanSlidingD2PeriReachCorr1{a} = nanmean(slidingD2Windows{a}(corr1Idx, :), 1);
        meanSlidingD2PeriReachCorr2{a} = nanmean(slidingD2Windows{a}(corr2Idx, :), 1);
        meanSlidingD2PeriReachErr1{a} = nanmean(slidingD2Windows{a}(err1Idx, :), 1);
        meanSlidingD2PeriReachErr2{a} = nanmean(slidingD2Windows{a}(err2Idx, :), 1);
        
        fprintf('Area %s: Sliding window d2 analysis completed\n', areas{a});
    end
end

% ==============================================     Plotting Results     ==============================================


% Create peri-reach plots for each area
if calculateSlidingWindowD2
    % 3 rows: original d2, sliding window d2, mrBr
    numRows = 3;
else
    % 2 rows: original d2, mrBr
    numRows = 2;
end

figure(300 + slidingWindowSize); clf;
% Prefer plotting on second screen if available
monitorPositions = get(0, 'MonitorPositions');
monitorTwo = monitorPositions(size(monitorPositions, 1), :);
if size(monitorPositions, 1) >= 2
    set(gcf, 'Position', monitorTwo);
else
    set(gcf, 'Position', monitorPositions(1, :));
end

% Use tight_subplot for layout with dynamic columns
numCols = length(areasToTest);
ha = tight_subplot(numRows, numCols, [0.08 0.04], [0.08 0.1], [0.06 0.04]);

colors = {'k', [0 0 .6], [.6 0 0], [0 0.6 0]};


% Compute global y-limits across areas (use mean traces; include errors if enabled)
allMeanVals = [];
for a = areasToTest
    % Block 1 (correct and error)
    if ~isempty(meanD2PeriReachCorr1{a})
        allMeanVals = [allMeanVals; meanD2PeriReachCorr1{a}(:)]; %#ok<AGROW>
    end
    if plotErrors && ~isempty(meanD2PeriReachErr1{a})
        allMeanVals = [allMeanVals; meanD2PeriReachErr1{a}(:)]; %#ok<AGROW>
    end
    % Block 2 (correct and error)
    if ~isempty(meanD2PeriReachCorr2{a})
        allMeanVals = [allMeanVals; meanD2PeriReachCorr2{a}(:)]; %#ok<AGROW>
    end
    if plotErrors && ~isempty(meanD2PeriReachErr2{a})
        allMeanVals = [allMeanVals; meanD2PeriReachErr2{a}(:)]; %#ok<AGROW>
    end
end
globalYMin = nanmin(allMeanVals);
globalYMax = nanmax(allMeanVals);
if isempty(globalYMin) || isempty(globalYMax) || isnan(globalYMin) || isnan(globalYMax)
    globalYMin = 0; globalYMax = 1;
end
% Add small padding
pad = 0.05 * (globalYMax - globalYMin + eps);
yLimCommon = [globalYMin - pad, globalYMax + pad];

% Compute global mrBr y-limits across areas (use mean traces; include errors if enabled)
allMrBrMeanVals = [];
for a = areasToTest
    if ~isempty(meanMrBrPeriReachCorr1{a})
        allMrBrMeanVals = [allMrBrMeanVals; meanMrBrPeriReachCorr1{a}(:)];
    end
    if ~isempty(meanMrBrPeriReachCorr2{a})
        allMrBrMeanVals = [allMrBrMeanVals; meanMrBrPeriReachCorr2{a}(:)];
    end
    if plotErrors
        if ~isempty(meanMrBrPeriReachErr1{a})
            allMrBrMeanVals = [allMrBrMeanVals; meanMrBrPeriReachErr1{a}(:)];
        end
        if ~isempty(meanMrBrPeriReachErr2{a})
            allMrBrMeanVals = [allMrBrMeanVals; meanMrBrPeriReachErr2{a}(:)];
        end
    end
end
mrBrYMin = nanmin(allMrBrMeanVals);
if isempty(mrBrYMin) || isnan(mrBrYMin)
    mrBrYMin = 0; % fallback
end
mrBrYMin = mrBrYMin - 0.05; % minimum minus 0.05
mrBrYMax = 1.05;            % maximum is 1 plus 0.05

% Compute global sliding window d2 y-limits across areas (if enabled)
if calculateSlidingWindowD2
    allSlidingD2MeanVals = [];
    for a = areasToTest
        % Block 1 (correct and error)
        if ~isempty(meanSlidingD2PeriReachCorr1{a})
            allSlidingD2MeanVals = [allSlidingD2MeanVals; meanSlidingD2PeriReachCorr1{a}(:)];
        end
        if plotErrors && ~isempty(meanSlidingD2PeriReachErr1{a})
            allSlidingD2MeanVals = [allSlidingD2MeanVals; meanSlidingD2PeriReachErr1{a}(:)];
        end
        % Block 2 (correct and error)
        if ~isempty(meanSlidingD2PeriReachCorr2{a})
            allSlidingD2MeanVals = [allSlidingD2MeanVals; meanSlidingD2PeriReachCorr2{a}(:)];
        end
        if plotErrors && ~isempty(meanSlidingD2PeriReachErr2{a})
            allSlidingD2MeanVals = [allSlidingD2MeanVals; meanSlidingD2PeriReachErr2{a}(:)];
        end
    end
    slidingD2YMin = nanmin(allSlidingD2MeanVals);
    slidingD2YMax = nanmax(allSlidingD2MeanVals);
    if isempty(slidingD2YMin) || isnan(slidingD2YMin) || isempty(slidingD2YMax) || isnan(slidingD2YMax)
        slidingD2YMin = 0; slidingD2YMax = 1;
    end
    % Add small padding
    pad = 0.05 * (slidingD2YMax - slidingD2YMin + eps);
    slidingD2YLim = [slidingD2YMin - pad, slidingD2YMax + pad];
end

for idx = 1:length(areasToTest)
    a = areasToTest(idx);
    % Top row: d2
    axes(ha(idx));
    hold on;

    % Determine if d2 data exist for this area (any mean trace non-NaN)
    hasD2 = (~isempty(meanD2PeriReachCorr1{a}) && any(~isnan(meanD2PeriReachCorr1{a}))) || ...
            (~isempty(meanD2PeriReachCorr2{a}) && any(~isnan(meanD2PeriReachCorr2{a}))) || ...
            (plotErrors && ~isempty(meanD2PeriReachErr1{a}) && any(~isnan(meanD2PeriReachErr1{a}))) || ...
            (plotErrors && ~isempty(meanD2PeriReachErr2{a}) && any(~isnan(meanD2PeriReachErr2{a})));

    % Compute x-limits safely
    if ~isempty(timeAxisPeriReach{a})
        xMin = min(timeAxisPeriReach{a});
        xMax = max(timeAxisPeriReach{a});
    else
        xMin = -1; xMax = 1; % fallback
    end

    if plotD2 && hasD2 && hasD2Area(a)
        % Calculate SEM for mean traces (use shared condition indices)
        semCorr1 = nanstd(d2Windows{a}(corr1Idx, :), 0, 1) / max(1, sqrt(sum(~all(isnan(d2Windows{a}(corr1Idx, :)), 2))));
        semCorr2 = nanstd(d2Windows{a}(corr2Idx, :), 0, 1) / max(1, sqrt(sum(~all(isnan(d2Windows{a}(corr2Idx, :)), 2))));
        if plotErrors
            semErr1 = nanstd(d2Windows{a}(err1Idx, :), 0, 1) / max(1, sqrt(sum(~all(isnan(d2Windows{a}(err1Idx, :)), 2))));
            semErr2 = nanstd(d2Windows{a}(err2Idx, :), 0, 1) / max(1, sqrt(sum(~all(isnan(d2Windows{a}(err2Idx, :)), 2))));
        end

        % Plot SEM ribbons first (behind the mean lines)
        baseColor = colorSpecToRGB(colors{a});
        colCorr1 = baseColor;                                % block 1
        colCorr2 = min(1, baseColor + 0.4);                  % lighten for block 2
        fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
             [meanD2PeriReachCorr1{a} + semCorr1, fliplr(meanD2PeriReachCorr1{a} - semCorr1)], ...
             colCorr1, 'FaceAlpha', 0.35, 'EdgeColor', 'none');
        fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
             [meanD2PeriReachCorr2{a} + semCorr2, fliplr(meanD2PeriReachCorr2{a} - semCorr2)], ...
             colCorr2, 'FaceAlpha', 0.2, 'EdgeColor', 'none');

        if plotErrors
            fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
                 [meanD2PeriReachErr1{a} + semErr1, fliplr(meanD2PeriReachErr1{a} - semErr1)], ...
                 colCorr1, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
            fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
                 [meanD2PeriReachErr2{a} + semErr2, fliplr(meanD2PeriReachErr2{a} - semErr2)], ...
                 colCorr2, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
        end

        % Plot mean d2 values per block
        hCorr1 = plot(timeAxisPeriReach{a}, meanD2PeriReachCorr1{a}, 'Color', colCorr1, 'LineWidth', 3, 'LineStyle', '-');
        hCorr2 = plot(timeAxisPeriReach{a}, meanD2PeriReachCorr2{a}, 'Color', colCorr2, 'LineWidth', 3, 'LineStyle', '-');
        if plotErrors
            hErr1 = plot(timeAxisPeriReach{a}, meanD2PeriReachErr1{a}, 'Color', colCorr1, 'LineWidth', 2, 'LineStyle', '--');
            hErr2 = plot(timeAxisPeriReach{a}, meanD2PeriReachErr2{a}, 'Color', colCorr2, 'LineWidth', 2, 'LineStyle', '--');
        end

        % Add vertical line at reach onset
        plot([0 0], ylim, 'k--', 'LineWidth', 2);
    end

    % Formatting (applies for both data-present and blank cases)
    xlabel('Time relative to reach onset (s)', 'FontSize', 12);
    ylabel('d2 Criticality', 'FontSize', 12);
    title(sprintf('%s - %s - Peri-Reach d2 Criticality (Window: %gs)', filePrefix, areas{a}, slidingWindowSize), 'FontSize', 14);
    grid on;
    xlim([xMin xMax]);
    xTicks = ceil(xMin):floor(xMax);
    if isempty(xTicks)
        xTicks = linspace(xMin, xMax, 5);
    end
    xticks(xTicks);
    xticklabels(string(xTicks));
    ylim(yLimCommon);
    yTicks = linspace(yLimCommon(1), yLimCommon(2), 5);
    yticks(yTicks);
    yticklabels(string(round(yTicks, 3)));

    % Legend only if data were plotted
    if plotD2 && hasD2 && hasD2Area(a)
        legs = [hCorr1 hCorr2];
        leg = {'d2 Correct B1', 'd2 Correct B2'};
        if plotErrors
            legs = [legs hErr1 hErr2];
            leg = [leg {'d2 Error B1', 'd2 Error B2'}];
        end
        legend(legs, leg, 'Location', 'best', 'FontSize', 9);
    end

    % Middle row: Sliding Window d2 (if enabled)
    if calculateSlidingWindowD2
        axes(ha(numCols + idx));
        hold on;

        % Determine if sliding window d2 data exist for this area
        hasSlidingD2 = (~isempty(meanSlidingD2PeriReachCorr1{a}) && any(~isnan(meanSlidingD2PeriReachCorr1{a}))) || ...
                       (~isempty(meanSlidingD2PeriReachCorr2{a}) && any(~isnan(meanSlidingD2PeriReachCorr2{a}))) || ...
                       (plotErrors && ~isempty(meanSlidingD2PeriReachErr1{a}) && any(~isnan(meanSlidingD2PeriReachErr1{a}))) || ...
                       (plotErrors && ~isempty(meanSlidingD2PeriReachErr2{a}) && any(~isnan(meanSlidingD2PeriReachErr2{a})));

        % Compute x-limits safely
        if ~isempty(timeAxisPeriReach{a})
            xMin = min(timeAxisPeriReach{a});
            xMax = max(timeAxisPeriReach{a});
        else
            xMin = -1; xMax = 1; % fallback
        end

        if plotD2 && hasSlidingD2
            % Calculate SEM for sliding window d2 mean traces (use shared condition indices)
            semSlidingCorr1 = nanstd(slidingD2Windows{a}(corr1Idx, :), 0, 1) / max(1, sqrt(sum(~all(isnan(slidingD2Windows{a}(corr1Idx, :)), 2))));
            semSlidingCorr2 = nanstd(slidingD2Windows{a}(corr2Idx, :), 0, 1) / max(1, sqrt(sum(~all(isnan(slidingD2Windows{a}(corr2Idx, :)), 2))));
            if plotErrors
                semSlidingErr1 = nanstd(slidingD2Windows{a}(err1Idx, :), 0, 1) / max(1, sqrt(sum(~all(isnan(slidingD2Windows{a}(err1Idx, :)), 2))));
                semSlidingErr2 = nanstd(slidingD2Windows{a}(err2Idx, :), 0, 1) / max(1, sqrt(sum(~all(isnan(slidingD2Windows{a}(err2Idx, :)), 2))));
            end

            % Plot SEM ribbons first (behind the mean lines)
            baseColor = colorSpecToRGB(colors{a});
            colCorr1 = baseColor;                                % block 1
            colCorr2 = min(1, baseColor + 0.4);                  % lighten for block 2
            fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
                 [meanSlidingD2PeriReachCorr1{a} + semSlidingCorr1, fliplr(meanSlidingD2PeriReachCorr1{a} - semSlidingCorr1)], ...
                 colCorr1, 'FaceAlpha', 0.35, 'EdgeColor', 'none');
            fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
                 [meanSlidingD2PeriReachCorr2{a} + semSlidingCorr2, fliplr(meanSlidingD2PeriReachCorr2{a} - semSlidingCorr2)], ...
                 colCorr2, 'FaceAlpha', 0.2, 'EdgeColor', 'none');

            if plotErrors
                fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
                     [meanSlidingD2PeriReachErr1{a} + semSlidingErr1, fliplr(meanSlidingD2PeriReachErr1{a} - semSlidingErr1)], ...
                     colCorr1, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
                fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
                     [meanSlidingD2PeriReachErr2{a} + semSlidingErr2, fliplr(meanSlidingD2PeriReachErr2{a} - semSlidingErr2)], ...
                     colCorr2, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
            end

            % Plot mean sliding window d2 values per block
            hSlidingCorr1 = plot(timeAxisPeriReach{a}, meanSlidingD2PeriReachCorr1{a}, 'Color', colCorr1, 'LineWidth', 3, 'LineStyle', '-');
            hSlidingCorr2 = plot(timeAxisPeriReach{a}, meanSlidingD2PeriReachCorr2{a}, 'Color', colCorr2, 'LineWidth', 3, 'LineStyle', '-');
            if plotErrors
                hSlidingErr1 = plot(timeAxisPeriReach{a}, meanSlidingD2PeriReachErr1{a}, 'Color', colCorr1, 'LineWidth', 2, 'LineStyle', '--');
                hSlidingErr2 = plot(timeAxisPeriReach{a}, meanSlidingD2PeriReachErr2{a}, 'Color', colCorr2, 'LineWidth', 2, 'LineStyle', '--');
            end

            % Add vertical line at reach onset
            plot([0 0], ylim, 'k--', 'LineWidth', 2);
        end

        % Formatting (applies for both data-present and blank cases)
        xlabel('Time relative to reach onset (s)', 'FontSize', 12);
        ylabel('Sliding Window d2', 'FontSize', 12);
        title(sprintf('%s - %s - Sliding Window d2 (Mean-Centered)', filePrefix, areas{a}), 'FontSize', 14);
        grid on;
        xlim([xMin xMax]);
        xTicks = ceil(xMin):floor(xMax);
        if isempty(xTicks)
            xTicks = linspace(xMin, xMax, 5);
        end
        xticks(xTicks);
        xticklabels(string(xTicks));
        if calculateSlidingWindowD2
            ylim(slidingD2YLim);
            yTicks = linspace(slidingD2YLim(1), slidingD2YLim(2), 5);
            yticks(yTicks);
            yticklabels(string(round(yTicks, 3)));
        end

        % Legend only if data were plotted
        if plotD2 && hasSlidingD2
            legs = [hSlidingCorr1 hSlidingCorr2];
            leg = {'Sliding d2 Correct B1', 'Sliding d2 Correct B2'};
            if plotErrors
                legs = [legs hSlidingErr1 hSlidingErr2];
                leg = [leg {'Sliding d2 Error B1', 'Sliding d2 Error B2'}];
            end
            legend(legs, leg, 'Location', 'best', 'FontSize', 9);
        end
    end

    % Bottom row: mrBr
    if calculateSlidingWindowD2
        axes(ha(2*numCols + idx));
    else
        axes(ha(numCols + idx));
    end
    hold on;

    % Determine if mrBr data exist for this area (any mean trace non-NaN)
    hasMr = (~isempty(meanMrBrPeriReachCorr1{a}) && any(~isnan(meanMrBrPeriReachCorr1{a}))) || ...
            (~isempty(meanMrBrPeriReachCorr2{a}) && any(~isnan(meanMrBrPeriReachCorr2{a}))) || ...
            (plotErrors && ~isempty(meanMrBrPeriReachErr1{a}) && any(~isnan(meanMrBrPeriReachErr1{a}))) || ...
            (plotErrors && ~isempty(meanMrBrPeriReachErr2{a}) && any(~isnan(meanMrBrPeriReachErr2{a})));

    % Compute x-limits safely
    if ~isempty(timeAxisPeriReach{a})
        xMin = min(timeAxisPeriReach{a}); xMax = max(timeAxisPeriReach{a});
    else
        xMin = -1; xMax = 1;
    end

    if plotMrBr && hasD2Area(a) && hasMr
        % Compute SEM for mrBr (use shared condition indices)
        baseColor = colorSpecToRGB(colors{a});
        colCorr1 = baseColor;
        colCorr2 = min(1, baseColor + 0.3);
        
        semMrCorr1 = nanstd(mrBrWindows{a}(corr1Idx, :), 0, 1) / max(1, sqrt(sum(~all(isnan(mrBrWindows{a}(corr1Idx, :)), 2))));
        semMrCorr2 = nanstd(mrBrWindows{a}(corr2Idx, :), 0, 1) / max(1, sqrt(sum(~all(isnan(mrBrWindows{a}(corr2Idx, :)), 2))));
        if plotErrors
            semMrErr1 = nanstd(mrBrWindows{a}(err1Idx, :), 0, 1) / max(1, sqrt(sum(~all(isnan(mrBrWindows{a}(err1Idx, :)), 2))));
            semMrErr2 = nanstd(mrBrWindows{a}(err2Idx, :), 0, 1) / max(1, sqrt(sum(~all(isnan(mrBrWindows{a}(err2Idx, :)), 2))));
        end

        % SEM ribbons
        fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
             [meanMrBrPeriReachCorr1{a} + semMrCorr1, fliplr(meanMrBrPeriReachCorr1{a} - semMrCorr1)], ...
             colCorr1, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
        fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
             [meanMrBrPeriReachCorr2{a} + semMrCorr2, fliplr(meanMrBrPeriReachCorr2{a} - semMrCorr2)], ...
             colCorr2, 'FaceAlpha', 0.4, 'EdgeColor', 'none');
        if plotErrors
            fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
                 [meanMrBrPeriReachErr1{a} + semMrErr1, fliplr(meanMrBrPeriReachErr1{a} - semMrErr1)], ...
                 colCorr1, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
            fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
                 [meanMrBrPeriReachErr2{a} + semMrErr2, fliplr(meanMrBrPeriReachErr2{a} - semMrErr2)], ...
                 colCorr2, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        end

        % Mean lines
        hMrCorr1 = plot(timeAxisPeriReach{a}, meanMrBrPeriReachCorr1{a}, 'Color', colCorr1, 'LineWidth', 3, 'LineStyle', '-');
        hMrCorr2 = plot(timeAxisPeriReach{a}, meanMrBrPeriReachCorr2{a}, 'Color', colCorr2, 'LineWidth', 3, 'LineStyle', '-.');
        if plotErrors
            hMrErr1 = plot(timeAxisPeriReach{a}, meanMrBrPeriReachErr1{a}, 'Color', colCorr1, 'LineWidth', 2.5, 'LineStyle', '--');
            hMrErr2 = plot(timeAxisPeriReach{a}, meanMrBrPeriReachErr2{a}, 'Color', colCorr2, 'LineWidth', 2.5, 'LineStyle', '--');
        end

        % Reference and onset lines
        yline(1, 'k:', 'LineWidth', 1.5);
        plot([0 0], ylim, 'k--', 'LineWidth', 2);
    end

    % Formatting (applies for both data-present and blank cases)
    xlabel('Time relative to reach onset (s)', 'FontSize', 12);
    ylabel('MR Branching Ratio', 'FontSize', 12);
    title(sprintf('%s - %s - Peri-Reach mrBr (Window: %gs)', filePrefix, areas{a}, slidingWindowSize), 'FontSize', 14);
    grid on;
    xlim([xMin xMax]);
    xTicks = ceil(xMin):floor(xMax); if isempty(xTicks), xTicks = linspace(xMin, xMax, 5); end
    xticks(xTicks); xticklabels(string(xTicks));
    ylim([mrBrYMin mrBrYMax]);
    set(gca, 'YTickLabelMode', 'auto');

    % Legend only if data were plotted
    if plotMrBr && hasD2Area(a) && hasMr
        if plotErrors
            legend([hMrCorr1 hMrCorr2 hMrErr1 hMrErr2], {'mrBr Correct B1','mrBr Correct B2','mrBr Error B1','mrBr Error B2'}, 'Location', 'best', 'FontSize', 9);
        else
            legend([hMrCorr1 hMrCorr2], {'mrBr Correct B1','mrBr Correct B2'}, 'Location', 'best', 'FontSize', 9);
        end
    end
end

if calculateSlidingWindowD2
    sgtitle(sprintf('%s - Peri-Reach d2 (top), Sliding Window d2 (middle), and mrBr (bottom) (Sliding Window: %gs)', filePrefix, slidingWindowSize), 'FontSize', 16);
else
    sgtitle(sprintf('%s - Peri-Reach d2 (top) and mrBr (bottom) (Sliding Window: %gs)', filePrefix, slidingWindowSize), 'FontSize', 16);
end

% Save combined figure (in same data-specific folder)
if calculateSlidingWindowD2
    filename = fullfile(saveDir, sprintf('%s_peri_reach_d2_sliding_d2_mrbr_win%gs.png', filePrefix, slidingWindowSize));
    fprintf('Saved peri-reach d2+sliding d2+mrBr plot to: %s\n', filename);
else
    filename = fullfile(saveDir, sprintf('%s_peri_reach_d2_mrbr_win%gs.png', filePrefix, slidingWindowSize));
    fprintf('Saved peri-reach d2+mrBr plot to: %s\n', filename);
end
exportgraphics(gcf, filename, 'Resolution', 300);

%% ==============================================     Summary Statistics     ==============================================

fprintf('\n=== Peri-Reach Analysis Summary ===\n');
for a = areasToTest
    fprintf('\nArea %s:\n', areas{a});
    fprintf('  Total reaches: %d\n', length(reachStartFrame{a}));
    fprintf('  Valid reaches: %d\n', sum(~all(isnan(d2Windows{a}), 2)));
    fprintf('  Mean d2 at reach onset: %.4f\n', meanD2PeriReach{a}(halfWindow + 1));
    fprintf('  Mean d2 pre-reach (-1s): %.4f\n', meanD2PeriReach{a}(halfWindow - round(1/optimalBinSize(a)) + 1));
    fprintf('  Mean d2 post-reach (+1s): %.4f\n', meanD2PeriReach{a}(halfWindow + round(1/optimalBinSize(a)) + 1));
end

fprintf('\nPeri-reach analysis complete!\n');




function rgb = colorSpecToRGB(c)
% Helper: convert MATLAB color char/spec to RGB triplet
    if ischar(c)
        switch c
            case 'k', rgb = [0 0 0];
            case 'b', rgb = [0 0 1];
            case 'r', rgb = [1 0 0];
            case 'g', rgb = [0 1 0];
            case 'c', rgb = [0 1 1];
            case 'm', rgb = [1 0 1];
            case 'y', rgb = [1 1 0];
            case 'w', rgb = [1 1 1];
            otherwise, rgb = [0 0 0];
        end
    else
        rgb = c;
    end
end
