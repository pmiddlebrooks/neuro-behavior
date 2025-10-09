%%
% N-Back Reach Criticality Analysis
% Loads results from criticality_reach_ar.m and analyzes d2 criticality values
% around reach onset times for trials following correct vs error reaches
%
% Variables:
%   results - loaded criticality analysis results
%   dataR - reach behavioral data
%   areas - brain areas to analyze
%   reachStartFrame - reach start times in frame units for each area
%   d2Windows - d2 criticality values in windows around each reach
%   meanD2PeriReach - mean d2 values across all reaches for each area

%% Load existing results if requested
slidingWindowSize = 4; % For d2, use a small window to try to optimize temporal resolution

% User-specified reach data file (should match the one used in criticality_reach_ar.m)
reachDataFile = fullfile(paths.dropPath, 'reach_data/AB2_01-May-2023 15_34_59_NeuroBeh.mat');
reachDataFile = fullfile(paths.dropPath, 'reach_data/AB2_11-May-2023 17_31_00_NeuroBeh.mat');
reachDataFile = fullfile(paths.dropPath, 'reach_data/AB2_28-Apr-2023 17_50_02_NeuroBeh.mat');
reachDataFile = fullfile(paths.dropPath, 'reach_data/AB2_30-May-2023 12_49_52_NeuroBeh.mat');
reachDataFile = fullfile(paths.dropPath, 'reach_data/AB6_02-Apr-2025 14_18_54_NeuroBeh.mat');
reachDataFile = fullfile(paths.dropPath, 'reach_data/AB6_03-Apr-2025 13_34_09_NeuroBeh.mat');
reachDataFile = fullfile(paths.dropPath, 'reach_data/AB6_27-Mar-2025 14_04_12_NeuroBeh.mat');
reachDataFile = fullfile(paths.dropPath, 'reach_data/AB6_29-Mar-2025 15_21_05_NeuroBeh.mat');
reachDataFile = fullfile(paths.dropPath, 'reach_data/Y4_06-Oct-2023 14_14_53_NeuroBeh.mat');

areasToTest = 1:4;

% Plotting toggles
plotD2 = true;     % plot d2 panels
plotMrBr = false;   % plot mrBr panels (only if d2 exists for area)

%
paths = get_paths;

% Determine save directory based on loaded data file name (same as criticality_reach_ar.m)
[~, dataBaseName, ~] = fileparts(reachDataFile);
saveDir = fullfile(paths.dropPath, 'reach_data', dataBaseName);
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
reachStop = dataR.R(:,2);
reachAmp = dataR.R(:,3); % Amplitude of each reach (distance from 0)

% Use Block(:,3) for reach classification
reachClass = dataR.Block(:,3);

% Define correct and error reaches (combining across blocks)
reachStartCorr = reachStart(ismember(reachClass, [2, 4])); % Correct reaches (block 1 and 2)
reachStartErr = reachStart(ismember(reachClass, [1, 3]));   % Error reaches (block 1 and 2)

% For n-back analysis, we need reaches that FOLLOW correct vs error trials
% Find reaches that come after correct trials (excluding first reach)
reachStartAfterCorr = [];
reachStartAfterErr = [];

for i = 2:length(reachStart)
    prevReachClass = reachClass(i-1);
    if ismember(prevReachClass, [2, 4]) % Previous reach was correct
        reachStartAfterCorr = [reachStartAfterCorr; reachStart(i)];
    elseif ismember(prevReachClass, [1, 3]) % Previous reach was error
        reachStartAfterErr = [reachStartAfterErr; reachStart(i)];
    end
end

% ==============================================     N-Back Reach Analysis     ==============================================

% Define window parameters
windowDurationSec = 40; % 40-second window around each reach
windowDurationFrames = cell(1, length(areas));
for a = areasToTest
    windowDurationFrames{a} = ceil(windowDurationSec / optimalBinSize(a));
end

% Initialize storage for peri-reach d2/mrBr values (following correct vs error)
d2WindowsAfterCorr = cell(1, length(areas));
d2WindowsAfterErr = cell(1, length(areas));
meanD2PeriReachAfterCorr = cell(1, length(areas));
meanD2PeriReachAfterErr = cell(1, length(areas));
mrBrWindowsAfterCorr = cell(1, length(areas));
mrBrWindowsAfterErr = cell(1, length(areas));
meanMrBrPeriReachAfterCorr = cell(1, length(areas));
meanMrBrPeriReachAfterErr = cell(1, length(areas));
timeAxisPeriReach = cell(1, length(areas));

% Flags indicating availability of data to analyze/plot per area
hasD2Area = false(1, length(areas));
hasMrBrArea = false(1, length(areas));

fprintf('\n=== N-Back Reach d2 Criticality Analysis ===\n');

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

    % Initialize arrays for this area
    numReachesAfterCorr = length(reachStartAfterCorr);
    numReachesAfterErr = length(reachStartAfterErr);
    halfWindow = floor(windowDurationFrames{a} / 2);

    % Create time axis for peri-reach window (centered on reach onset)
    timeAxisPeriReach{a} = (-halfWindow:halfWindow) * optimalBinSize(a);

    % Initialize storage for all reach windows (following correct vs error)
    d2WindowsAfterCorr{a} = nan(numReachesAfterCorr, windowDurationFrames{a} + 1);
    d2WindowsAfterErr{a} = nan(numReachesAfterErr, windowDurationFrames{a} + 1);
    mrBrWindowsAfterCorr{a} = nan(numReachesAfterCorr, windowDurationFrames{a} + 1);
    mrBrWindowsAfterErr{a} = nan(numReachesAfterErr, windowDurationFrames{a} + 1);

    % Extract around each reach following a correct trial
    validAfterCorr = 0;
    for r = 1:numReachesAfterCorr
        reachTime = reachStartAfterCorr(r)/1000; % adjust if needed based on units
        [~, closestIdx] = min(abs(timePoints - reachTime));
        winStart = closestIdx - halfWindow;
        winEnd = closestIdx + halfWindow;
        if winStart >= 1 && winEnd <= length(timePoints)
            if hasD2Area(a)
                d2WindowsAfterCorr{a}(r, :) = d2Values(winStart:winEnd);
            end
            if hasMrBrArea(a)
                mrBrWindowsAfterCorr{a}(r, :) = mrValues(winStart:winEnd);
            end
            validAfterCorr = validAfterCorr + 1;
        else
            d2WindowsAfterCorr{a}(r, :) = nan(1, windowDurationFrames{a} + 1);
            mrBrWindowsAfterCorr{a}(r, :) = nan(1, windowDurationFrames{a} + 1);
        end
    end

    % Extract around each reach following an error trial
    validAfterErr = 0;
    for r = 1:numReachesAfterErr
        reachTime = reachStartAfterErr(r)/1000; % adjust if needed based on units
        [~, closestIdx] = min(abs(timePoints - reachTime));
        winStart = closestIdx - halfWindow;
        winEnd = closestIdx + halfWindow;
        if winStart >= 1 && winEnd <= length(timePoints)
            if hasD2Area(a)
                d2WindowsAfterErr{a}(r, :) = d2Values(winStart:winEnd);
            end
            if hasMrBrArea(a)
                mrBrWindowsAfterErr{a}(r, :) = mrValues(winStart:winEnd);
            end
            validAfterErr = validAfterErr + 1;
        else
            d2WindowsAfterErr{a}(r, :) = nan(1, windowDurationFrames{a} + 1);
            mrBrWindowsAfterErr{a}(r, :) = nan(1, windowDurationFrames{a} + 1);
        end
    end

    % Calculate mean d2/mrBr values across all valid reaches (conditional)
    % d2 means (exists by guard above)
    meanD2PeriReachAfterCorr{a} = nanmean(d2WindowsAfterCorr{a}, 1);
    meanD2PeriReachAfterErr{a} = nanmean(d2WindowsAfterErr{a}, 1);

    % mrBr means (computed only if available; plotted only if d2 exists)
    if hasMrBrArea(a)
        meanMrBrPeriReachAfterCorr{a} = nanmean(mrBrWindowsAfterCorr{a}, 1);
        meanMrBrPeriReachAfterErr{a} = nanmean(mrBrWindowsAfterErr{a}, 1);
    end

    fprintf('Area %s: After Corr=%d, After Err=%d valid reaches\n', areas{a}, validAfterCorr, validAfterErr);
end

% ==============================================     Plotting Results     ==============================================

% Create peri-reach plots for each area: 2 (metrics) x numAreas
figure(400 + slidingWindowSize); clf;
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
ha = tight_subplot(2, numCols, [0.08 0.04], [0.08 0.1], [0.06 0.04]);

colors = {'k', [0 0 .6], [.6 0 0], [0 0.6 0]};

% Compute global y-limits across areas (use mean traces)
allMeanVals = [];
for a = areasToTest
    if ~isempty(meanD2PeriReachAfterCorr{a})
        allMeanVals = [allMeanVals; meanD2PeriReachAfterCorr{a}(:)]; %#ok<AGROW>
    end
    if ~isempty(meanD2PeriReachAfterErr{a})
        allMeanVals = [allMeanVals; meanD2PeriReachAfterErr{a}(:)]; %#ok<AGROW>
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

% Compute global mrBr y-limits across areas (use mean traces)
allMrBrMeanVals = [];
for a = areasToTest
    if ~isempty(meanMrBrPeriReachAfterCorr{a})
        allMrBrMeanVals = [allMrBrMeanVals; meanMrBrPeriReachAfterCorr{a}(:)];
    end
    if ~isempty(meanMrBrPeriReachAfterErr{a})
        allMrBrMeanVals = [allMrBrMeanVals; meanMrBrPeriReachAfterErr{a}(:)];
    end
end
mrBrYMin = nanmin(allMrBrMeanVals);
if isempty(mrBrYMin) || isnan(mrBrYMin)
    mrBrYMin = 0; % fallback
end
mrBrYMin = mrBrYMin - 0.05; % minimum minus 0.05
mrBrYMax = 1.05;            % maximum is 1 plus 0.05

for idx = 1:length(areasToTest)
    a = areasToTest(idx);
    % Top row: d2
    axes(ha(idx));
    hold on;

    % Determine if d2 data exist for this area (any mean trace non-NaN)
    hasD2 = (~isempty(meanD2PeriReachAfterCorr{a}) && any(~isnan(meanD2PeriReachAfterCorr{a}))) || ...
            (~isempty(meanD2PeriReachAfterErr{a}) && any(~isnan(meanD2PeriReachAfterErr{a})));

    % Compute x-limits safely
    if ~isempty(timeAxisPeriReach{a})
        xMin = min(timeAxisPeriReach{a});
        xMax = max(timeAxisPeriReach{a});
    else
        xMin = -1; xMax = 1; % fallback
    end

    if plotD2 && hasD2 && hasD2Area(a)
        % Calculate SEM for mean traces
        semAfterCorr = nanstd(d2WindowsAfterCorr{a}, 0, 1) / max(1, sqrt(sum(~all(isnan(d2WindowsAfterCorr{a}), 2))));
        semAfterErr = nanstd(d2WindowsAfterErr{a}, 0, 1) / max(1, sqrt(sum(~all(isnan(d2WindowsAfterErr{a}), 2))));

        % Plot SEM ribbons first (behind the mean lines)
        baseColor = colorSpecToRGB(colors{a});
        colAfterCorr = baseColor;                                % after correct
        % colAfterErr = min(1, baseColor + 0.4);                  % lighten for after error
        colAfterErr = colAfterCorr;                  % lighten for after error
        fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
             [meanD2PeriReachAfterCorr{a} + semAfterCorr, fliplr(meanD2PeriReachAfterCorr{a} - semAfterCorr)], ...
             colAfterCorr, 'FaceAlpha', 0.35, 'EdgeColor', 'none');
        fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
             [meanD2PeriReachAfterErr{a} + semAfterErr, fliplr(meanD2PeriReachAfterErr{a} - semAfterErr)], ...
             colAfterErr, 'FaceAlpha', 0.2, 'EdgeColor', 'none');

        % Plot mean d2 values
        hAfterCorr = plot(timeAxisPeriReach{a}, meanD2PeriReachAfterCorr{a}, 'Color', colAfterCorr, 'LineWidth', 3, 'LineStyle', '-');
        hAfterErr = plot(timeAxisPeriReach{a}, meanD2PeriReachAfterErr{a}, 'Color', colAfterErr, 'LineWidth', 3, 'LineStyle', '--');

        % Add vertical line at reach onset
        plot([0 0], ylim, 'k--', 'LineWidth', 2);
    end

    % Formatting (applies for both data-present and blank cases)
    xlabel('Time relative to reach onset (s)', 'FontSize', 12);
    ylabel('d2 Criticality', 'FontSize', 12);
    title(sprintf('%s - %s - N-Back Reach d2 Criticality (Window: %gs)', filePrefix, areas{a}, slidingWindowSize), 'FontSize', 14);
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
        legend([hAfterCorr hAfterErr], {'d2 After Correct', 'd2 After Error'}, 'Location', 'best', 'FontSize', 9);
    end

    % Bottom row: mrBr
    axes(ha(numCols + idx));
    hold on;

    % Determine if mrBr data exist for this area (any mean trace non-NaN)
    hasMr = (~isempty(meanMrBrPeriReachAfterCorr{a}) && any(~isnan(meanMrBrPeriReachAfterCorr{a}))) || ...
            (~isempty(meanMrBrPeriReachAfterErr{a}) && any(~isnan(meanMrBrPeriReachAfterErr{a})));

    % Compute x-limits safely
    if ~isempty(timeAxisPeriReach{a})
        xMin = min(timeAxisPeriReach{a}); xMax = max(timeAxisPeriReach{a});
    else
        xMin = -1; xMax = 1;
    end

    if plotMrBr && hasD2Area(a) && hasMr
        % Compute SEM for mrBr
        baseColor = colorSpecToRGB(colors{a});
        colAfterCorr = baseColor;
        colAfterErr = min(1, baseColor + 0.3);
        semMrAfterCorr = nanstd(mrBrWindowsAfterCorr{a}, 0, 1) / max(1, sqrt(sum(~all(isnan(mrBrWindowsAfterCorr{a}), 2))));
        semMrAfterErr = nanstd(mrBrWindowsAfterErr{a}, 0, 1) / max(1, sqrt(sum(~all(isnan(mrBrWindowsAfterErr{a}), 2))));

        % SEM ribbons
        fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
             [meanMrBrPeriReachAfterCorr{a} + semMrAfterCorr, fliplr(meanMrBrPeriReachAfterCorr{a} - semMrAfterCorr)], ...
             colAfterCorr, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
        fill([timeAxisPeriReach{a}, fliplr(timeAxisPeriReach{a})], ...
             [meanMrBrPeriReachAfterErr{a} + semMrAfterErr, fliplr(meanMrBrPeriReachAfterErr{a} - semMrAfterErr)], ...
             colAfterErr, 'FaceAlpha', 0.4, 'EdgeColor', 'none');

        % Mean lines
        hMrAfterCorr = plot(timeAxisPeriReach{a}, meanMrBrPeriReachAfterCorr{a}, 'Color', colAfterCorr, 'LineWidth', 3, 'LineStyle', '-');
        hMrAfterErr = plot(timeAxisPeriReach{a}, meanMrBrPeriReachAfterErr{a}, 'Color', colAfterErr, 'LineWidth', 3, 'LineStyle', '--');

        % Reference and onset lines
        yline(1, 'k:', 'LineWidth', 1.5);
        plot([0 0], ylim, 'k--', 'LineWidth', 2);
    end

    % Formatting (applies for both data-present and blank cases)
    xlabel('Time relative to reach onset (s)', 'FontSize', 12);
    ylabel('MR Branching Ratio', 'FontSize', 12);
    title(sprintf('%s - %s - N-Back Reach mrBr (Window: %gs)', filePrefix, areas{a}, slidingWindowSize), 'FontSize', 14);
    grid on;
    xlim([xMin xMax]);
    xTicks = ceil(xMin):floor(xMax); if isempty(xTicks), xTicks = linspace(xMin, xMax, 5); end
    xticks(xTicks); xticklabels(string(xTicks));
    ylim([mrBrYMin mrBrYMax]);
    set(gca, 'YTickLabelMode', 'auto');

    % Legend only if data were plotted
    if plotMrBr && hasD2Area(a) && hasMr
        legend([hMrAfterCorr hMrAfterErr], {'mrBr After Correct','mrBr After Error'}, 'Location', 'best', 'FontSize', 9);
    end
end

sgtitle(sprintf('%s - N-Back Reach d2 (top) and mrBr (bottom) (Sliding Window: %gs)', filePrefix, slidingWindowSize), 'FontSize', 16);

% Save combined figure (in same data-specific folder)
filename = fullfile(saveDir, sprintf('%s_nback_reach_d2_mrbr_win%gs.png', filePrefix, slidingWindowSize));
exportgraphics(gcf, filename, 'Resolution', 300);
fprintf('Saved n-back reach d2+mrBr plot to: %s\n', filename);

%% ==============================================     Summary Statistics     ==============================================

fprintf('\n=== N-Back Reach Analysis Summary ===\n');
for a = areasToTest
    fprintf('\nArea %s:\n', areas{a});
    fprintf('  Reaches after correct: %d\n', sum(~all(isnan(d2WindowsAfterCorr{a}), 2)));
    fprintf('  Reaches after error: %d\n', sum(~all(isnan(d2WindowsAfterErr{a}), 2)));
    if ~isempty(meanD2PeriReachAfterCorr{a})
        fprintf('  Mean d2 after correct at reach onset: %.4f\n', meanD2PeriReachAfterCorr{a}(halfWindow + 1));
    end
    if ~isempty(meanD2PeriReachAfterErr{a})
        fprintf('  Mean d2 after error at reach onset: %.4f\n', meanD2PeriReachAfterErr{a}(halfWindow + 1));
    end
end

fprintf('\nN-back reach analysis complete!\n');

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
