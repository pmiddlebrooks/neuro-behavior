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
slidingWindowSize = 3;% For d2, use a small window to try to optimize temporal resolution

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
% reachDataFile = fullfile(paths.reachDataPath, 'makeSpikes.mat');
% reachDataFile = fullfile(paths.dropPath, 'reach_task/data/Y4_06-Oct-2023 14_14_53_NeuroBeh.mat');

areasToTest = 1:4;

% Toggle plotting error reaches (single traces and mean)
plotErrors = true;
% Plotting toggles
plotD2 = true;     % plot d2 panels
analyzeMrBr = false;   % plot mrBr panels (only if d2 exists for area)

% NEW: Sliding window d2 calculation options
analyzeMeanCenteredD2 = true;  % Set to true to perform sliding window d2 on mean-centered popActivity
d2Window = 36;  % Window duration in seconds for collecting popActivity per reach

% NEW: Modulation analysis options
analyzeModulation = true;  % Set to true to plot modulated vs unmodulated (combines all conditions)

%
paths = get_paths;

% Determine save directory based on loaded data file name (same as criticality_reach_ar.m)
[~, dataBaseName, ~] = fileparts(reachDataFile);
saveDir = fullfile(paths.reachResultsPath, dataBaseName);
resultsPathWin = fullfile(saveDir, sprintf('criticality_sliding_window_ar_win%d.mat', slidingWindowSize));

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
optimalBinSize = results.optimalBinSize;
d2Rea = results.d2;
mrBrRea = results.mrBr;
startS = results.startS;

% Extract modulation results if available
if isfield(results, 'analyzeModulation') && results.analyzeModulation && isfield(results, 'modulationResults')
    modulationResults = results.modulationResults;
    fprintf('Loaded modulation analysis results.\n');

    % Display modulation statistics
    if analyzeModulation
        fprintf('\n=== Modulation Statistics ===\n');
        for a = areasToTest
            if ~isempty(modulationResults{a})
                fprintf('Area %s: %d/%d neurons modulated (%.1f%%)\n', areas{a}, ...
                    sum(modulationResults{a}.isModulated), length(modulationResults{a}.neuronIds), ...
                    100*sum(modulationResults{a}.isModulated)/length(modulationResults{a}.neuronIds));
            end
        end
    end
    
    % Extract modulated/unmodulated bin sizes if available
    if isfield(results, 'optimalBinSizeModulated') && isfield(results, 'optimalBinSizeUnmodulated')
        optimalBinSizeModulated = results.optimalBinSizeModulated;
        optimalBinSizeUnmodulated = results.optimalBinSizeUnmodulated;
        fprintf('Loaded modulated/unmodulated optimal bin sizes.\n');
    else
        optimalBinSizeModulated = nan(1, length(areas));
        optimalBinSizeUnmodulated = nan(1, length(areas));
    end
else
    modulationResults = cell(1, length(areas));
    optimalBinSizeModulated = nan(1, length(areas));
    optimalBinSizeUnmodulated = nan(1, length(areas));
    fprintf('No modulation analysis results found in saved data.\n');
    if analyzeModulation
        fprintf('Warning: analyzeModulation is set to true, but no modulation data exists. Setting analyzeModulation to false.\n');
        analyzeModulation = false;
    end
end

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
windowDurationSec = 30; % 3-second window around each reach
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

    if hasMrBrArea(a) && analyzeMrBr
        meanMrBrPeriReachCorr1{a} = nanmean(mrBrWindows{a}(corr1Idx, :), 1);
        meanMrBrPeriReachCorr2{a} = nanmean(mrBrWindows{a}(corr2Idx, :), 1);
        meanMrBrPeriReachErr1{a} = nanmean(mrBrWindows{a}(err1Idx, :), 1);
        meanMrBrPeriReachErr2{a} = nanmean(mrBrWindows{a}(err2Idx, :), 1);
    end

end

% ==============================================     Sliding Window D2 Analysis     ==============================================

if analyzeMeanCenteredD2
    [slidingD2Windows, meanSlidingD2PeriReachCorr1, meanSlidingD2PeriReachCorr2, meanSlidingD2PeriReachErr1, meanSlidingD2PeriReachErr2] = calculate_sliding_window_d2(results, areasToTest, areas, optimalBinSize, reachStart, windowDurationFrames, d2Window, corr1Idx, corr2Idx, err1Idx, err2Idx);
end

% ==============================================     Modulation Analysis     ==============================================

if analyzeModulation
    fprintf('\n=== Modulated vs Unmodulated Peri-Reach Analysis ===\n');
    
    % Initialize storage for modulated/unmodulated peri-reach windows
    d2ModulatedWindows = cell(1, length(areas));
    d2UnmodulatedWindows = cell(1, length(areas));
    meanD2ModulatedPeriReach = cell(1, length(areas));
    meanD2UnmodulatedPeriReach = cell(1, length(areas));
    timeAxisModulated = cell(1, length(areas));
    timeAxisUnmodulated = cell(1, length(areas));
    
    for a = areasToTest
        fprintf('\nProcessing area %s for modulated vs unmodulated...\n', areas{a});
        
        % Check if modulation data exists
        if isempty(modulationResults{a})
            fprintf('Skipping area %s - no modulation data\n', areas{a});
            continue;
        end
        
        % Get d2 values for modulated and unmodulated populations
        d2Mod = results.d2Modulated{a};
        d2Unmod = results.d2Unmodulated{a};
        startSMod = results.startSModulated{a};
        startSUnmod = results.startSUnmodulated{a};
        
        if isempty(d2Mod) || isempty(d2Unmod)
            fprintf('Skipping area %s - no modulated/unmodulated d2 data\n', areas{a});
            continue;
        end
        
        % Get the appropriate bin sizes for each population
        if ~isnan(optimalBinSizeModulated(a)) && optimalBinSizeModulated(a) > 0
            binSizeMod = optimalBinSizeModulated(a);
        else
            binSizeMod = optimalBinSize(a);
        end
        
        if ~isnan(optimalBinSizeUnmodulated(a)) && optimalBinSizeUnmodulated(a) > 0
            binSizeUnmod = optimalBinSizeUnmodulated(a);
        else
            binSizeUnmod = optimalBinSize(a);
        end
        
        % Calculate window parameters for each population
        windowDurationFramesMod = ceil(windowDurationSec / binSizeMod);
        windowDurationFramesUnmod = ceil(windowDurationSec / binSizeUnmod);
        halfWindowMod = floor(windowDurationFramesMod / 2);
        halfWindowUnmod = floor(windowDurationFramesUnmod / 2);
        
        % Create time axis for each population (centered on reach onset, 0 at center)
        timeAxisModulated{a} = (-halfWindowMod:halfWindowMod) * binSizeMod;
        timeAxisUnmodulated{a} = (-halfWindowUnmod:halfWindowUnmod) * binSizeUnmod;
        
        % Initialize storage for each population using its window size
        d2ModulatedWindows{a} = nan(totalReaches, windowDurationFramesMod + 1);
        d2UnmodulatedWindows{a} = nan(totalReaches, windowDurationFramesUnmod + 1);
        
        % Extract around each reach (same approach as normal d2)
        for r = 1:totalReaches
            reachTime = reachStart(r)/1000; % Convert to seconds
            
            % Modulated population
            [~, closestIdxMod] = min(abs(startSMod - reachTime));
            winStartMod = closestIdxMod - halfWindowMod;
            winEndMod = closestIdxMod + halfWindowMod;
            
            if winStartMod >= 1 && winEndMod <= length(startSMod)
                d2ModulatedWindows{a}(r, :) = d2Mod(winStartMod:winEndMod);
            else
                d2ModulatedWindows{a}(r, :) = nan(1, windowDurationFramesMod + 1);
            end
            
            % Unmodulated population
            [~, closestIdxUnmod] = min(abs(startSUnmod - reachTime));
            winStartUnmod = closestIdxUnmod - halfWindowUnmod;
            winEndUnmod = closestIdxUnmod + halfWindowUnmod;
            
            if winStartUnmod >= 1 && winEndUnmod <= length(startSUnmod)
                d2UnmodulatedWindows{a}(r, :) = d2Unmod(winStartUnmod:winEndUnmod);
            else
                d2UnmodulatedWindows{a}(r, :) = nan(1, windowDurationFramesUnmod + 1);
        end
    end

        % Calculate mean across all reaches (combining all conditions)
        meanD2ModulatedPeriReach{a} = nanmean(d2ModulatedWindows{a}, 1);
        meanD2UnmodulatedPeriReach{a} = nanmean(d2UnmodulatedWindows{a}, 1);
        
        fprintf('Area %s: Modulated vs unmodulated analysis completed\n', areas{a});
    end
end

% ==============================================     Plotting Results     ==============================================


% Create peri-reach plots for each area
% Determine number of rows based on enabled analyses
numRows = 1; % Always plot original d2 (row 1)
if analyzeMeanCenteredD2
    numRows = numRows + 1; % Mean-centered d2 (row 2)
end
if analyzeModulation
    numRows = numRows + 1; % Modulated vs unmodulated (row 3)
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

% Initialize modulated/unmodulated variables if not created
if ~exist('d2ModulatedWindows', 'var')
    d2ModulatedWindows = cell(1, length(areas));
    d2UnmodulatedWindows = cell(1, length(areas));
    meanD2ModulatedPeriReach = cell(1, length(areas));
    meanD2UnmodulatedPeriReach = cell(1, length(areas));
    timeAxisModulated = cell(1, length(areas));
    timeAxisUnmodulated = cell(1, length(areas));
end

% Calculate plot y-limits
[yLimCommon, slidingD2YLim] = calculate_d2_plot_ylimits(meanD2PeriReachCorr1, meanD2PeriReachCorr2, meanD2PeriReachErr1, meanD2PeriReachErr2, meanSlidingD2PeriReachCorr1, meanSlidingD2PeriReachCorr2, meanSlidingD2PeriReachErr1, meanSlidingD2PeriReachErr2, meanD2ModulatedPeriReach, meanD2UnmodulatedPeriReach, areasToTest, plotErrors, analyzeMeanCenteredD2, analyzeModulation);

% Calculate mrBr y-limits separately (only if plotting mrBr)
if analyzeMrBr
    mrBrYLim = calculate_mrbr_plot_ylimits(mrBrWindows, corr1Idx, corr2Idx, err1Idx, err2Idx, areasToTest, plotErrors, analyzeMrBr);
else
    mrBrYLim = [0, 1]; % Default fallback
end

% Create all plots using function
create_peri_reach_plots(d2Windows, mrBrWindows, slidingD2Windows, d2ModulatedWindows, d2UnmodulatedWindows, timeAxisPeriReach, timeAxisModulated, timeAxisUnmodulated, meanD2PeriReachCorr1, meanD2PeriReachCorr2, meanD2PeriReachErr1, meanD2PeriReachErr2, meanSlidingD2PeriReachCorr1, meanSlidingD2PeriReachCorr2, meanSlidingD2PeriReachErr1, meanSlidingD2PeriReachErr2, meanD2ModulatedPeriReach, meanD2UnmodulatedPeriReach, areasToTest, areas, slidingWindowSize, filePrefix, plotD2, analyzeMrBr, plotErrors, analyzeMeanCenteredD2, analyzeModulation, yLimCommon, mrBrYLim, slidingD2YLim, corr1Idx, corr2Idx, err1Idx, err2Idx, saveDir, colors, ha, numCols, hasD2Area);

% Generate title based on enabled analyses
if analyzeMeanCenteredD2 && analyzeModulation && analyzeMrBr
    sgtitle(sprintf('%s - d2, Sliding d2, Mod/Unmod, mrBr (Window: %gs)', filePrefix, slidingWindowSize), 'FontSize', 16);
elseif analyzeMeanCenteredD2 && analyzeModulation
    sgtitle(sprintf('%s - d2, Sliding d2, Mod/Unmod (Window: %gs)', filePrefix, slidingWindowSize), 'FontSize', 16);
elseif analyzeModulation && analyzeMrBr
    sgtitle(sprintf('%s - d2, Mod/Unmod, mrBr (Window: %gs)', filePrefix, slidingWindowSize), 'FontSize', 16);
elseif analyzeMeanCenteredD2 && analyzeMrBr
    sgtitle(sprintf('%s - Peri-Reach d2 (top), Sliding Window d2 (middle), and mrBr (bottom) (Sliding Window: %gs)', filePrefix, slidingWindowSize), 'FontSize', 16);
elseif analyzeMeanCenteredD2
    sgtitle(sprintf('%s - Peri-Reach d2 (top) and Sliding Window d2 (bottom) (Sliding Window: %gs)', filePrefix, slidingWindowSize), 'FontSize', 16);
elseif analyzeModulation
    sgtitle(sprintf('%s - Peri-Reach d2 (top) and Modulated vs Unmodulated (Window: %gs)', filePrefix, slidingWindowSize), 'FontSize', 16);
elseif analyzeMrBr
    sgtitle(sprintf('%s - Peri-Reach d2 (top) and mrBr (bottom) (Sliding Window: %gs)', filePrefix, slidingWindowSize), 'FontSize', 16);
else
    sgtitle(sprintf('%s - Peri-Reach d2 (Sliding Window: %gs)', filePrefix, slidingWindowSize), 'FontSize', 16);
end

% Save combined figure (in same data-specific folder)
if analyzeMeanCenteredD2 && analyzeModulation && analyzeMrBr
    filename = fullfile(saveDir, sprintf('%s_peri_reach_d2_sliding_mod_unmod_mrbr_win%gs.png', filePrefix, slidingWindowSize));
    fprintf('Saved peri-reach d2+sliding+mod/unmod+mrBr plot to: %s\n', filename);
elseif analyzeMeanCenteredD2 && analyzeModulation
    filename = fullfile(saveDir, sprintf('%s_peri_reach_d2_sliding_mod_unmod_win%gs.png', filePrefix, slidingWindowSize));
    fprintf('Saved peri-reach d2+sliding+mod/unmod plot to: %s\n', filename);
elseif analyzeModulation && analyzeMrBr
    filename = fullfile(saveDir, sprintf('%s_peri_reach_d2_mod_unmod_mrbr_win%gs.png', filePrefix, slidingWindowSize));
    fprintf('Saved peri-reach d2+mod/unmod+mrBr plot to: %s\n', filename);
elseif analyzeModulation
    filename = fullfile(saveDir, sprintf('%s_peri_reach_d2_mod_unmod_win%gs.png', filePrefix, slidingWindowSize));
    fprintf('Saved peri-reach d2+mod/unmod plot to: %s\n', filename);
elseif analyzeMeanCenteredD2 && analyzeMrBr
    filename = fullfile(saveDir, sprintf('%s_peri_reach_d2_sliding_d2_mrbr_win%gs.png', filePrefix, slidingWindowSize));
    fprintf('Saved peri-reach d2+sliding d2+mrBr plot to: %s\n', filename);
elseif analyzeMeanCenteredD2
    filename = fullfile(saveDir, sprintf('%s_peri_reach_d2_sliding_d2_win%gs.png', filePrefix, slidingWindowSize));
    fprintf('Saved peri-reach d2+sliding d2 plot to: %s\n', filename);
elseif analyzeMrBr
    filename = fullfile(saveDir, sprintf('%s_peri_reach_d2_mrbr_win%gs.png', filePrefix, slidingWindowSize));
    fprintf('Saved peri-reach d2+mrBr plot to: %s\n', filename);
else
    filename = fullfile(saveDir, sprintf('%s_peri_reach_d2_win%gs.png', filePrefix, slidingWindowSize));
    fprintf('Saved peri-reach d2 plot to: %s\n', filename);
end
exportgraphics(gcf, filename, 'Resolution', 300);

%% ==============================================     Summary Statistics     ==============================================

fprintf('\n=== Peri-Reach Analysis Summary ===\n');
for a = areasToTest
    fprintf('\nArea %s:\n', areas{a});
    fprintf('  Total reaches: %d\n', totalReaches);
    fprintf('  Valid reaches: %d\n', sum(~all(isnan(d2Windows{a}), 2)));
    if hasD2Area(a) && ~isempty(meanD2PeriReachCorr1{a})
        halfWindow = floor(windowDurationFrames{a} / 2);
        fprintf('  Mean d2 at reach onset: %.4f\n', meanD2PeriReachCorr1{a}(halfWindow + 1));
        fprintf('  Mean d2 pre-reach (-1s): %.4f\n', meanD2PeriReachCorr1{a}(halfWindow - round(1/optimalBinSize(a)) + 1));
        fprintf('  Mean d2 post-reach (+1s): %.4f\n', meanD2PeriReachCorr1{a}(halfWindow + round(1/optimalBinSize(a)) + 1));
    end
end

fprintf('\nPeri-reach analysis complete!\n');

%% =============================    Function Definitions    =============================

function [slidingD2Windows, meanSlidingD2PeriReachCorr1, meanSlidingD2PeriReachCorr2, meanSlidingD2PeriReachErr1, meanSlidingD2PeriReachErr2] = ...
    calculate_sliding_window_d2(results, areasToTest, areas, optimalBinSize, reachStart, windowDurationFrames, d2Window, corr1Idx, corr2Idx, err1Idx, err2Idx)
% Calculate sliding window d2 analysis on mean-centered population activity
fprintf('\n=== D2 Analysis on Mean-Centered Population Activity ===\n');

% Get parameters from results
pOrder = results.params.pOrder;
critType = results.params.critType;

% Initialize storage for sliding window d2 results
slidingD2Windows = cell(1, length(areas));
meanSlidingD2PeriReachCorr1 = cell(1, length(areas));
meanSlidingD2PeriReachCorr2 = cell(1, length(areas));
meanSlidingD2PeriReachErr1 = cell(1, length(areas));
meanSlidingD2PeriReachErr2 = cell(1, length(areas));

totalReaches = length(reachStart);

for a = areasToTest
    fprintf('\nProcessing sliding window d2 for area %s...\n', areas{a});

    % Get population activity for this area
    popActivityFull = results.popActivity{a};
    if isempty(popActivityFull)
        fprintf('Skipping area %s due to missing population activity data.\n', areas{a});
        continue;
    end

    % Get time points and bin size for this area
    binSize = optimalBinSize(a);
    timePoints = 0 : binSize : length(popActivityFull)*binSize - binSize; % real time points (not sliding window centers)

    % Calculate window parameters
    d2WindowFrames = ceil(d2Window / binSize);
    halfD2Window = floor(d2WindowFrames / 2);

    % Calculate sliding window parameters
    stepSamples = round(results.d2StepSize(a) / binSize);
    winSamples = round(results.d2WindowSize(a) / binSize);

    % Initialize storage for all reach windows
    slidingD2Windows{a} = nan(totalReaches, windowDurationFrames{a} + 1);

    % Collect all popActivity windows for mean-centering
    allPopActivityWindows = [];
    validReachIndices = [];

    % First pass: collect all valid popActivity windows for mean-centering
    for r = 1:totalReaches
        reachTime = reachStart(r)/1000; % Convert to seconds
        [~, closestIdx] = min(abs(timePoints - reachTime));
        winStart = closestIdx - halfD2Window;
        winEnd = closestIdx + halfD2Window-1;

        if winStart >= 1 && winEnd <= length(popActivityFull)
            popWindow = popActivityFull(winStart:winEnd);
            allPopActivityWindows = [allPopActivityWindows; popWindow'];
            validReachIndices = [validReachIndices, r];
        end
    end

    if isempty(allPopActivityWindows)
        fprintf('No valid popActivity windows found for area %s\n', areas{a});
        continue;
    end

    % Calculate mean popActivity across all valid reaches
    meanPopActivity = mean(allPopActivityWindows, 1);

    % Mean-center each popActivity window
    meanCenteredPopActivity = allPopActivityWindows - meanPopActivity;

    % Second pass: perform sliding window d2 analysis on each mean-centered reach
    for reachIdx = 1:length(validReachIndices)
        r = validReachIndices(reachIdx);
        reachTime = reachStart(r)/1000;
        [~, closestIdx] = min(abs(timePoints - reachTime));

        % Get the mean-centered popActivity for this reach
        meanCenteredPop = meanCenteredPopActivity(reachIdx, :);

        % Perform sliding window d2 analysis within the d2Window
        % The sliding window should slide from half-window-size into the d2Window
        % until half-window-size remains, with results centered on d2Window center

        % Calculate how many sliding windows we can fit within the d2Window
        % Start when we have a full sliding window, end when we have a full sliding window remaining
        startOffset = floor(winSamples / 2);  % Half sliding window size
        endOffset = floor(winSamples / 2);    % Half sliding window size

        % Calculate the range of sliding window positions within d2Window
        maxStartPos = length(meanCenteredPop) - winSamples - endOffset + 1;
        minStartPos = startOffset + 1;

        if maxStartPos >= minStartPos
            % Calculate number of sliding windows that fit
            numSlidingWindows = floor((maxStartPos - minStartPos) / stepSamples) + 1;

            % Initialize array to store d2 values at their proper positions
            slidingD2Values = nan(1, d2WindowFrames);

            for w = 1:numSlidingWindows
                % Calculate start position for this sliding window
                startIdx = minStartPos + (w - 1) * stepSamples;
                endIdx = startIdx + winSamples - 1;

                if endIdx <= length(meanCenteredPop)
                    wPopActivity = meanCenteredPop(startIdx:endIdx);

                    % Calculate d2 using Yule-Walker
                    try
                        [varphi, ~] = myYuleWalker3(wPopActivity, pOrder);
                        d2Value = getFixedPointDistance2(pOrder, critType, varphi);

                        % Store d2 value at the center position of the sliding window
                        centerPos = startIdx + floor(winSamples / 2);
                        if centerPos >= 1 && centerPos <= length(slidingD2Values)
                            slidingD2Values(centerPos) = d2Value;
                        end
                    catch
                        % Leave as NaN if calculation fails
                    end
                end
            end
        else
            slidingD2Values = nan(1, d2WindowFrames);
        end

        % Map sliding window d2 values to peri-reach window
        % Both windows should be centered on the reach time (closestIdx)

        halfWindow = floor(windowDurationFrames{a} / 2);
        winStartPeri = closestIdx - halfWindow;
        winEndPeri = closestIdx + halfWindow;

        if winStartPeri >= 1 && winEndPeri <= length(timePoints)
            % The sliding window d2 analysis is performed within the d2Window
            % which is centered on the reach time. We need to map these values
            % to the peri-reach window which is also centered on the reach time.

            % Calculate the center position of the d2Window (where reach onset is)
            d2WindowCenter = floor(d2WindowFrames / 2) + 1;

            % Map sliding window d2 values to peri-reach window
            % The center of the d2Window (reach onset) should align with the center of the peri-reach window
            for d2Idx = 1:length(slidingD2Values)
                if ~isnan(slidingD2Values(d2Idx))
                    % Calculate offset from d2Window center
                    offsetFromCenter = d2Idx - d2WindowCenter;

                    % Map to peri-reach window center + offset
                    periIdx = floor(windowDurationFrames{a} / 2) + 1 + offsetFromCenter;

                    % Ensure the index is within the peri-reach window bounds
                    if periIdx >= 1 && periIdx <= size(slidingD2Windows{a}, 2)
                        slidingD2Windows{a}(r, periIdx) = slidingD2Values(d2Idx);
                    end
                end
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

function [yLimCommon, slidingD2YLim] = calculate_d2_plot_ylimits(meanD2PeriReachCorr1, meanD2PeriReachCorr2, meanD2PeriReachErr1, meanD2PeriReachErr2, meanSlidingD2PeriReachCorr1, meanSlidingD2PeriReachCorr2, meanSlidingD2PeriReachErr1, meanSlidingD2PeriReachErr2, meanD2ModulatedPeriReach, meanD2UnmodulatedPeriReach, areasToTest, plotErrors, analyzeMeanCenteredD2, analyzeModulation)
% Calculate unified global y-limits for all d2 plots (normal, mean-centered, and modulated/unmodulated)

% Collect all d2 values for unified y-limits
allD2Vals = [];

% Add normal d2 values
for a = areasToTest
    % Block 1 (correct and error)
    if ~isempty(meanD2PeriReachCorr1{a})
        allD2Vals = [allD2Vals; meanD2PeriReachCorr1{a}(:)]; %#ok<AGROW>
    end
    if plotErrors && ~isempty(meanD2PeriReachErr1{a})
        allD2Vals = [allD2Vals; meanD2PeriReachErr1{a}(:)]; %#ok<AGROW>
    end
    % Block 2 (correct and error)
    if ~isempty(meanD2PeriReachCorr2{a})
        allD2Vals = [allD2Vals; meanD2PeriReachCorr2{a}(:)]; %#ok<AGROW>
    end
    if plotErrors && ~isempty(meanD2PeriReachErr2{a})
        allD2Vals = [allD2Vals; meanD2PeriReachErr2{a}(:)]; %#ok<AGROW>
    end
end

% Add mean-centered d2 values (if enabled)
if analyzeMeanCenteredD2
    for a = areasToTest
        % Block 1 (correct and error)
        if ~isempty(meanSlidingD2PeriReachCorr1{a})
            allD2Vals = [allD2Vals; meanSlidingD2PeriReachCorr1{a}(:)];
        end
        if plotErrors && ~isempty(meanSlidingD2PeriReachErr1{a})
            allD2Vals = [allD2Vals; meanSlidingD2PeriReachErr1{a}(:)];
        end
        % Block 2 (correct and error)
        if ~isempty(meanSlidingD2PeriReachCorr2{a})
            allD2Vals = [allD2Vals; meanSlidingD2PeriReachCorr2{a}(:)];
        end
        if plotErrors && ~isempty(meanSlidingD2PeriReachErr2{a})
            allD2Vals = [allD2Vals; meanSlidingD2PeriReachErr2{a}(:)];
        end
    end
end

% Add modulated/unmodulated d2 values (if enabled)
if analyzeModulation
    for a = areasToTest
        if ~isempty(meanD2ModulatedPeriReach{a})
            allD2Vals = [allD2Vals; meanD2ModulatedPeriReach{a}(:)];
        end
        if ~isempty(meanD2UnmodulatedPeriReach{a})
            allD2Vals = [allD2Vals; meanD2UnmodulatedPeriReach{a}(:)];
        end
    end
end

% Calculate unified y-limits
globalYMin = nanmin(allD2Vals);
globalYMax = nanmax(allD2Vals);
if isempty(globalYMin) || isempty(globalYMax) || isnan(globalYMin) || isnan(globalYMax)
    globalYMin = 0; globalYMax = 1;
end
% Add small padding
pad = 0.05 * (globalYMax - globalYMin + eps);
unifiedYLim = [globalYMin - pad, globalYMax + pad];

% Use the same y-limits for both normal d2 and sliding window d2
yLimCommon = unifiedYLim;
slidingD2YLim = unifiedYLim;
end

function mrBrYLim = calculate_mrbr_plot_ylimits(mrBrWindows, corr1Idx, corr2Idx, err1Idx, err2Idx, areasToTest, plotErrors, analyzeMrBr)
% Calculate global y-limits for mrBr plots

allMrBrMeanVals = [];
for a = areasToTest
    if ~isempty(mrBrWindows{a})
        % Calculate mean mrBr values for each condition
        meanMrBrCorr1 = nanmean(mrBrWindows{a}(corr1Idx, :), 1);
        meanMrBrCorr2 = nanmean(mrBrWindows{a}(corr2Idx, :), 1);

        if ~isempty(meanMrBrCorr1)
            allMrBrMeanVals = [allMrBrMeanVals; meanMrBrCorr1(:)];
        end
        if ~isempty(meanMrBrCorr2)
            allMrBrMeanVals = [allMrBrMeanVals; meanMrBrCorr2(:)];
        end

        if plotErrors
            meanMrBrErr1 = nanmean(mrBrWindows{a}(err1Idx, :), 1);
            meanMrBrErr2 = nanmean(mrBrWindows{a}(err2Idx, :), 1);

            if ~isempty(meanMrBrErr1)
                allMrBrMeanVals = [allMrBrMeanVals; meanMrBrErr1(:)];
            end
            if ~isempty(meanMrBrErr2)
                allMrBrMeanVals = [allMrBrMeanVals; meanMrBrErr2(:)];
            end
        end
    end
end

mrBrYMin = nanmin(allMrBrMeanVals);
if isempty(mrBrYMin) || isnan(mrBrYMin)
    mrBrYMin = 0; % fallback
end
mrBrYMin = mrBrYMin - 0.05; % minimum minus 0.05
mrBrYMax = 1.05;            % maximum is 1 plus 0.05
mrBrYLim = [mrBrYMin, mrBrYMax];
end

function create_peri_reach_plots(d2Windows, mrBrWindows, slidingD2Windows, d2ModulatedWindows, d2UnmodulatedWindows, timeAxisPeriReach, timeAxisModulated, timeAxisUnmodulated, meanD2PeriReachCorr1, meanD2PeriReachCorr2, meanD2PeriReachErr1, meanD2PeriReachErr2, meanSlidingD2PeriReachCorr1, meanSlidingD2PeriReachCorr2, meanSlidingD2PeriReachErr1, meanSlidingD2PeriReachErr2, meanD2ModulatedPeriReach, meanD2UnmodulatedPeriReach, areasToTest, areas, slidingWindowSize, filePrefix, plotD2, analyzeMrBr, plotErrors, analyzeMeanCenteredD2, analyzeModulation, yLimCommon, mrBrYLim, slidingD2YLim, corr1Idx, corr2Idx, err1Idx, err2Idx, saveDir, colors, ha, numCols, hasD2Area)
% Create all peri-reach plots

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
    title(sprintf('%s - Peri-Reach d2 Criticality (Window: %gs)', areas{a}, slidingWindowSize), 'FontSize', 14);
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
    if analyzeMeanCenteredD2
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
        title(sprintf('%s - Mean-Centered d2', areas{a}), 'FontSize', 14);
        grid on;
        xlim([xMin xMax]);
        xTicks = ceil(xMin):floor(xMax);
        if isempty(xTicks)
            xTicks = linspace(xMin, xMax, 5);
        end
        xticks(xTicks);
        xticklabels(string(xTicks));
        if analyzeMeanCenteredD2
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

    % Third row: Modulated vs Unmodulated (if enabled)
    if analyzeModulation
        % Determine which row based on whether mean-centered d2 is enabled
        if analyzeMeanCenteredD2
            axes(ha(2*numCols + idx)); % Row 3 (after mean-centered d2)
        else
            axes(ha(numCols + idx)); % Row 2 (if mean-centered d2 not enabled)
        end
        hold on;

        % Check if data exists
        hasModData = ~isempty(meanD2ModulatedPeriReach{a}) && any(~isnan(meanD2ModulatedPeriReach{a}));
        hasUnmodData = ~isempty(meanD2UnmodulatedPeriReach{a}) && any(~isnan(meanD2UnmodulatedPeriReach{a}));

        if hasModData || hasUnmodData
            % Plot modulated using its own time axis
            if hasModData && ~isempty(timeAxisModulated{a})
                plot(timeAxisModulated{a}, meanD2ModulatedPeriReach{a}, 'r-', 'LineWidth', 3, 'DisplayName', 'Modulated');
            end

            % Plot unmodulated using its own time axis
            if hasUnmodData && ~isempty(timeAxisUnmodulated{a})
                plot(timeAxisUnmodulated{a}, meanD2UnmodulatedPeriReach{a}, 'b-', 'LineWidth', 3, 'DisplayName', 'Unmodulated');
            end

            plot([0 0], ylim, 'k--', 'LineWidth', 2);
            legend('Location', 'best', 'FontSize', 10);
        end

        xlabel('Time relative to reach onset (s)', 'FontSize', 12);
        ylabel('d2', 'FontSize', 12);
        title(sprintf('%s - Modulated vs Unmodulated', areas{a}), 'FontSize', 14);
        grid on;
        
        % Use x-limits based on the broader time range
        if hasModData && ~isempty(timeAxisModulated{a})
            xMin = min(timeAxisModulated{a});
            xMax = max(timeAxisModulated{a});
        elseif hasUnmodData && ~isempty(timeAxisUnmodulated{a})
            xMin = min(timeAxisUnmodulated{a});
            xMax = max(timeAxisUnmodulated{a});
        end
        if hasUnmodData && ~isempty(timeAxisUnmodulated{a})
            xMin = min(xMin, min(timeAxisUnmodulated{a}));
            xMax = max(xMax, max(timeAxisUnmodulated{a}));
        end
        
        xlim([xMin xMax]);
        xTicks = ceil(xMin):floor(xMax); if isempty(xTicks), xTicks = linspace(xMin, xMax, 5); end
        xticks(xTicks); xticklabels(string(xTicks));
        ylim(yLimCommon);
        set(gca, 'YTickLabelMode', 'auto');
    end

    % Bottom row: mrBr (skip if modulation is enabled as mrBr is not being plotted)
    if ~analyzeModulation && analyzeMrBr
        if analyzeMeanCenteredD2
            axes(ha(2*numCols + idx));
        else
            axes(ha(numCols + idx));
        end
    hold on;

    % Determine if mrBr data exist for this area (any mean trace non-NaN)
        if analyzeMrBr
            % Calculate mrBr means internally
            meanMrBrPeriReachCorr1 = nanmean(mrBrWindows{a}(corr1Idx, :), 1);
            meanMrBrPeriReachCorr2 = nanmean(mrBrWindows{a}(corr2Idx, :), 1);
            meanMrBrPeriReachErr1 = nanmean(mrBrWindows{a}(err1Idx, :), 1);
            meanMrBrPeriReachErr2 = nanmean(mrBrWindows{a}(err2Idx, :), 1);

            hasMr = (~isempty(meanMrBrPeriReachCorr1) && any(~isnan(meanMrBrPeriReachCorr1))) || ...
                (~isempty(meanMrBrPeriReachCorr2) && any(~isnan(meanMrBrPeriReachCorr2))) || ...
                (plotErrors && ~isempty(meanMrBrPeriReachErr1) && any(~isnan(meanMrBrPeriReachErr1))) || ...
                (plotErrors && ~isempty(meanMrBrPeriReachErr2) && any(~isnan(meanMrBrPeriReachErr2)));

    % Compute x-limits safely
    if ~isempty(timeAxisPeriReach{a})
        xMin = min(timeAxisPeriReach{a}); xMax = max(timeAxisPeriReach{a});
    else
        xMin = -1; xMax = 1;
    end

            if analyzeMrBr && hasD2Area(a) && hasMr
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
                    [meanMrBrPeriReachCorr1 + semMrCorr1, fliplr(meanMrBrPeriReachCorr1 - semMrCorr1)], ...
             colCorr1, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
        fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
                    [meanMrBrPeriReachCorr2 + semMrCorr2, fliplr(meanMrBrPeriReachCorr2 - semMrCorr2)], ...
             colCorr2, 'FaceAlpha', 0.4, 'EdgeColor', 'none');
        if plotErrors
            fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
                        [meanMrBrPeriReachErr1 + semMrErr1, fliplr(meanMrBrPeriReachErr1 - semMrErr1)], ...
                 colCorr1, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
            fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
                        [meanMrBrPeriReachErr2 + semMrErr2, fliplr(meanMrBrPeriReachErr2 - semMrErr2)], ...
                 colCorr2, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        end

        % Mean lines
                hMrCorr1 = plot(timeAxisPeriReach{a}, meanMrBrPeriReachCorr1, 'Color', colCorr1, 'LineWidth', 3, 'LineStyle', '-');
                hMrCorr2 = plot(timeAxisPeriReach{a}, meanMrBrPeriReachCorr2, 'Color', colCorr2, 'LineWidth', 3, 'LineStyle', '-.');
        if plotErrors
                    hMrErr1 = plot(timeAxisPeriReach{a}, meanMrBrPeriReachErr1, 'Color', colCorr1, 'LineWidth', 2.5, 'LineStyle', '--');
                    hMrErr2 = plot(timeAxisPeriReach{a}, meanMrBrPeriReachErr2, 'Color', colCorr2, 'LineWidth', 2.5, 'LineStyle', '--');
        end

        % Reference and onset lines
        yline(1, 'k:', 'LineWidth', 1.5);
        plot([0 0], ylim, 'k--', 'LineWidth', 2);
    end

    % Formatting (applies for both data-present and blank cases)
    xlabel('Time relative to reach onset (s)', 'FontSize', 12);
    ylabel('MR Branching Ratio', 'FontSize', 12);
            title(sprintf('%s - Peri-Reach mrBr (Window: %gs)', areas{a}, slidingWindowSize), 'FontSize', 14);
    grid on;
    xlim([xMin xMax]);
    xTicks = ceil(xMin):floor(xMax); if isempty(xTicks), xTicks = linspace(xMin, xMax, 5); end
    xticks(xTicks); xticklabels(string(xTicks));
            ylim(mrBrYLim);
    set(gca, 'YTickLabelMode', 'auto');

    % Legend only if data were plotted
            if analyzeMrBr && hasD2Area(a) && hasMr
        if plotErrors
            legend([hMrCorr1 hMrCorr2 hMrErr1 hMrErr2], {'mrBr Correct B1','mrBr Correct B2','mrBr Error B1','mrBr Error B2'}, 'Location', 'best', 'FontSize', 9);
        else
            legend([hMrCorr1 hMrCorr2], {'mrBr Correct B1','mrBr Correct B2'}, 'Location', 'best', 'FontSize', 9);
        end
    end
end
    end
end
end



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
