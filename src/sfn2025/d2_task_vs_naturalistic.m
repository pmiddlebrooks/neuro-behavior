%% D2_TASK_VS_NATURALISTIC - Compare d2 criticality measures between reach task and naturalistic data
%
% This script loads criticality analysis results for both reach task and
% naturalistic data, then creates a comparison figure showing:
%   1. Ethogram of behavior labels (top subplot)
%   2. d2 time series for both conditions with reach start markers (bottom subplot)
%
% Variables:
%   areaIdx - Brain area index (1=M23, 2=M56, 3=DS, 4=VS)
%   resultsReach - Loaded results from reach task criticality analysis
%   resultsNat - Loaded results from naturalistic criticality analysis
%   reachStart - Reach start times in seconds
%   bhvIDReach - Behavior labels for reach data
%   bhvIDNat - Behavior labels for naturalistic data
%   colorsReach - Color mapping for reach behaviors
%   colorsNat - Color mapping for naturalistic behaviors

%% User-specified parameters
areaIdx = 2; % Brain area: 1=M23, 2=M56, 3=DS, 4=VS

% Sliding window size (should match the one used in criticality_sliding_window_ar.m)
slidingWindowSize = 20; % seconds

% PCA flag (should match the one used in criticality_sliding_window_ar.m)
usePCA = false;  % Set to true if PCA was used in the analysis

% Setup paths
paths = get_paths;
areas = {'M23', 'M56', 'DS', 'VS'};

% Create filename suffix based on PCA flag
if usePCA
    filenameSuffix = '_pca';
else
    filenameSuffix = '';
end

% Reach data file (should match the one used in criticality_sliding_window_ar.m)
        % sessionName =  'AB2_28-Apr-2023 17_50_02_NeuroBeh.mat';
        % sessionName =  'AB2_01-May-2023 15_34_59_NeuroBeh.mat';
        % sessionName =  'AB2_11-May-2023 17_31_00_NeuroBeh.mat';
        % sessionName =  'AB2_30-May-2023 12_49_52_NeuroBeh.mat';
        % sessionName =  'AB6_27-Mar-2025 14_04_12_NeuroBeh.mat';
        % sessionName =  'AB6_29-Mar-2025 15_21_05_NeuroBeh.mat';
        % sessionName =  'AB6_02-Apr-2025 14_18_54_NeuroBeh.mat';
        % sessionName =  'AB6_03-Apr-2025 13_34_09_NeuroBeh.mat';
        % sessionName =  'Y4_06-Oct-2023 14_14_53_NeuroBeh.mat';
        % sessionName =  'Y15_26-Aug-2025 12_24_22_NeuroBeh.mat';
        % sessionName =  'Y15_27-Aug-2025 14_02_21_NeuroBeh.mat';
        % sessionName =  'Y15_28-Aug-2025 19_47_07_NeuroBeh.mat';
        % sessionName =  'Y17_20-Aug-2025 17_34_48_NeuroBeh.mat';
reachDataFile = fullfile(paths.reachDataPath, sessionName);
[~, dataBaseName, ~] = fileparts(reachDataFile);

% Naturalistic data parameters (should match criticality_sliding_window_ar.m)
animal = 'ag25290';
sessionBhv = '112321_1';

% ================================    Load criticality results
% Load reach criticality results
fprintf('\n=== Loading Reach Criticality Results ===\n');
reachSaveDir = fullfile(paths.dropPath, 'reach_task/results', dataBaseName);
reachResultsPath = fullfile(reachSaveDir, sprintf('criticality_sliding_window_ar%s_win%d.mat', filenameSuffix, slidingWindowSize));

if ~exist(reachResultsPath, 'file')
    error('Reach results file not found: %s\nRun criticality_sliding_window_ar.m for reach data first.', reachResultsPath);
end

resultsReach = load(reachResultsPath);
resultsReach = resultsReach.results;
fprintf('Loaded reach results for area %s\n', areas{areaIdx});

% Load naturalistic criticality results
fprintf('\n=== Loading Naturalistic Criticality Results ===\n');
natSaveDir = fullfile(paths.dropPath, 'criticality/results');
natResultsPath = fullfile(natSaveDir, sprintf('criticality_sliding_window_ar%s_win%d.mat', filenameSuffix, slidingWindowSize));

if ~exist(natResultsPath, 'file')
    error('Naturalistic results file not found: %s\nRun criticality_sliding_window_ar.m for naturalistic data first.', natResultsPath);
end

resultsNat = load(natResultsPath);
resultsNat = resultsNat.results;
fprintf('Loaded naturalistic results for area %s\n', areas{areaIdx});


% ================================    Load  data
% Load reach data for behavior labels and reach start times
fprintf('\n=== Loading Reach Data ===\n');
dataR = load(reachDataFile);
reachStart = dataR.R(:,1) / 1000; % Convert from ms to seconds
    reachClass = dataR.Block(:,3);
    startBlock2 = reachStart(find(ismember(reachClass, [3 4]), 1));

% Get reach behavior labels
opts = neuro_behavior_options;
opts.frameSize = 0.001; % seconds (fine resolution for ethogram)
opts.collectStart = 0;
opts.collectEnd = round(min(dataR.R(end,1) + 5000, max(dataR.CSV(:,1)*1000)) / 1000);
bhvOpts = struct();
bhvOpts.frameSize = opts.frameSize;
bhvOpts.collectStart = opts.collectStart;
bhvOpts.collectEnd = opts.collectEnd;
bhvIDReach = define_reach_bhv_labels(reachDataFile, bhvOpts);

% Create time axis for reach behavior labels
timeAxisReachBhv = (0:length(bhvIDReach)-1) * opts.frameSize;

% Reach behavior colors (matching svm_umap_figures.m)
behaviorsReach = {'pre-reach', 'reach', 'pre-reward', 'reward', 'post-reward', 'intertrial'};
nBehaviorsReach = length(behaviorsReach);
try
    func = @sRGB_to_OKLab;
    cOpts.exc = [0,0,0];
    cOpts.Lmax = .8;
    colorsReach = maxdistcolor(nBehaviorsReach, func, cOpts);
    colorsReach(end,:) = [0 0 0]; % intertrial is black
catch
    colorsReach = lines(nBehaviorsReach);
    colorsReach(end,:) = [0 0 0];
end

% Load naturalistic behavior data
fprintf('\n=== Loading Naturalistic Behavior Data ===\n');
optsNat = neuro_behavior_options;
optsNat.collectStart = 0 * 60 * 60; % seconds
optsNat.collectEnd = 45 * 60; % seconds
optsNat.frameSize = opts.frameSize; % Use same fine resolution

bhvDataPath = strcat(paths.bhvDataPath, 'animal_', animal, '/');
bhvFileName = ['behavior_labels_', animal, '_', sessionBhv, '.csv'];
optsNat.dataPath = bhvDataPath;
optsNat.fileName = bhvFileName;
dataBhvNat = load_data(optsNat, 'behavior');

% Create bhvID for naturalistic data (matching get_standard_data.m approach)
nFrame = ceil(optsNat.collectEnd / optsNat.frameSize);
dataBhvNat.StartFrame = 1 + round(dataBhvNat.StartTime / optsNat.frameSize);
bhvIDNat = int8(zeros(nFrame, 1));
for i = 1:size(dataBhvNat, 1) - 1
    iInd = dataBhvNat.StartFrame(i) : dataBhvNat.StartFrame(i+1) - 1;
    bhvIDNat(iInd) = dataBhvNat.ID(i);
end
if ~isempty(dataBhvNat)
    iInd = dataBhvNat.StartFrame(end):nFrame;
    if ~isempty(iInd)
        bhvIDNat(iInd) = dataBhvNat.ID(end);
    end
end

% Create time axis for naturalistic behavior labels
timeAxisNatBhv = (0:length(bhvIDNat)-1) * optsNat.frameSize;

% Naturalistic behavior colors (matching svm_umap_figures.m)
codesNat = unique(dataBhvNat.ID);
try
    colorsNat = colors_for_behaviors(codesNat);
catch
    colorsNat = lines(max(codesNat) + 2);
end

% ================================    Find maximum d2 across all areas and conditions
% Loop through all areas to find the global maximum d2 value
fprintf('\n=== Finding Maximum d2 Across All Areas and Conditions ===\n');
maxD2 = -inf;
for iArea = 1:length(areas)
    % Extract d2 for reach condition
    d2ReachAll = resultsReach.d2{iArea};
    if ~isempty(d2ReachAll) && ~all(isnan(d2ReachAll))
        maxD2Reach = max(d2ReachAll(~isnan(d2ReachAll)));
        if maxD2Reach > maxD2
            maxD2 = maxD2Reach;
        end
    end
    
    % Extract d2 for naturalistic condition
    d2NatAll = resultsNat.d2{iArea};
    if ~isempty(d2NatAll) && ~all(isnan(d2NatAll))
        maxD2Nat = max(d2NatAll(~isnan(d2NatAll)));
        if maxD2Nat > maxD2
            maxD2 = maxD2Nat;
        end
    end
end
fprintf('Maximum d2 value across all areas and conditions: %.4f\n', maxD2);


% ================================    Analysis   ================================
% Extract d2 data for specified area
d2Reach = resultsReach.d2{areaIdx};
startSReach = resultsReach.startS{areaIdx};
d2Nat = resultsNat.d2{areaIdx};
startSNat = resultsNat.startS{areaIdx};

% Find common time range
timeStart = min([min(startSReach), min(startSNat), 0]);
timeEnd = max([max(startSReach), max(startSNat), max(timeAxisReachBhv), max(timeAxisNatBhv)]);

saveDir = fullfile(paths.dropPath, 'sfn2025', 'd2_comparison');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end


% ================================    Create d2 Time Series Figure
fprintf('\n=== Creating d2 Time Series Figure ===\n');
figure(713); clf;
set(gcf, 'Units', 'pixels');
monitorPositions = get(0, 'MonitorPositions');
if size(monitorPositions, 1) >= 2
    targetMonitor = monitorPositions(size(monitorPositions, 1), :);
else
    targetMonitor = monitorPositions(1, :);
end
set(gcf, 'Position', [targetMonitor(1), targetMonitor(2), targetMonitor(3), targetMonitor(4)*.5]);

hold on;

% Plot naturalistic d2
if ~isempty(d2Nat) && ~all(isnan(d2Nat))
    plot(startSNat, d2Nat, 'r-', 'LineWidth', 3, 'DisplayName', 'spontaneous d2');
end

% Plot reach d2
if ~isempty(d2Reach) && ~all(isnan(d2Reach))
    plot(startSReach, d2Reach, 'b-', 'LineWidth', 3, 'DisplayName', 'task d2');
end
                        xline(startBlock2, 'Color', [0 .8 .2], 'LineWidth', 3);

% Plot reach start markers (filled green circles)
reachStartInRange = reachStart(reachStart >= timeStart & reachStart <= timeEnd);
if ~isempty(reachStartInRange)
    % Get d2 values at reach start times (interpolate if needed)
    d2AtReach = interp1(startSReach, d2Reach, reachStartInRange, 'linear', 'extrap');
    scatter(reachStartInRange, d2AtReach, 80, 'g', 'filled', 'o', ...
        'DisplayName', 'task Start', 'MarkerEdgeColor', 'k', 'LineWidth', 1);
end

xlim([timeStart, timeEnd]);
ylim([0 max([d2Reach(:); d2Nat(:)])]);
xlabel('Time (s)', 'FontSize', 12);
ylabel('d2', 'FontSize', 12);
title(sprintf('%s - d2 Criticality Comparison', areas{areaIdx}), 'FontSize', 12);
legend('Location', 'best');
grid on;

set(gca, 'FontSize', 18)
% Save d2 time series figure
saveFileD2 = fullfile(saveDir, sprintf('d2_task_vs_naturalistic_%s%s_win%d_d2_%s.eps', areas{areaIdx}, filenameSuffix, slidingWindowSize, sessionName));
% exportgraphics(gcf, saveFileD2, 'ContentType', 'vector');
saveFileD2 = fullfile(saveDir, sprintf('d2_task_vs_naturalistic_%s%s_win%d_d2_%s.png', areas{areaIdx}, filenameSuffix, slidingWindowSize, sessionName));
    exportgraphics(gcf, saveFileD2, 'Resolution', 300);
fprintf('Saved d2 time series figure to: %s\n', saveFileD2);

 %% ================================    Create Ethogram Figure
% fprintf('\n=== Creating Ethogram Figure ===\n');
% figure(711); clf;
% set(gcf, 'Units', 'pixels');
% monitorPositions = get(0, 'MonitorPositions');
% if size(monitorPositions, 1) >= 2
%     targetMonitor = monitorPositions(size(monitorPositions, 1), :);
% else
%     targetMonitor = monitorPositions(1, :);
% end
% set(gcf, 'Position', [targetMonitor(1), targetMonitor(2), targetMonitor(3), targetMonitor(4)/3]);
% 
% hold on;
% 
% % Plot reach behavior ethogram
% for bhvCode = 1:6
%     bhvMask = (bhvIDReach == bhvCode);
%     if any(bhvMask)
%         % Create color blocks by finding contiguous regions
%         diffMask = diff([0; bhvMask; 0]);
%         starts = find(diffMask == 1);
%         ends = find(diffMask == -1) - 1;
% 
%         for b = 1:length(starts)
%             xStart = timeAxisReachBhv(starts(b));
%             xEnd = timeAxisReachBhv(ends(b));
%             fill([xStart, xEnd, xEnd, xStart], [0, 0, 1, 1], colorsReach(bhvCode, :), ...
%                 'EdgeColor', 'none', 'DisplayName', behaviorsReach{bhvCode});
%         end
%     end
% end
% 
% % Plot naturalistic behavior ethogram (offset vertically)
% for bhvCode = unique(bhvIDNat)'
%     if bhvCode >= 0 % Skip invalid labels
%         bhvMask = (bhvIDNat == bhvCode);
%         if any(bhvMask)
%             % Create color blocks by finding contiguous regions
%             diffMask = diff([0; bhvMask; 0]);
%             starts = find(diffMask == 1);
%             ends = find(diffMask == -1) - 1;
% 
%             % Find color index
%             colorIdx = find(codesNat == bhvCode, 1);
%             if ~isempty(colorIdx) && colorIdx <= size(colorsNat, 1)
%                 plotColor = colorsNat(colorIdx, :);
%             else
%                 plotColor = [0.5, 0.5, 0.5]; % Gray for unmapped
%             end
% 
%             for b = 1:length(starts)
%                 xStart = timeAxisNatBhv(starts(b));
%                 xEnd = timeAxisNatBhv(ends(b));
%                 fill([xStart, xEnd, xEnd, xStart], [1, 1, 2, 2], plotColor, ...
%                     'EdgeColor', 'none');
%             end
%         end
%     end
% end
% 
% xlim([timeStart, timeEnd]);
% ylim([0, 2]);
% ylabel('Behavior', 'FontSize', 12);
% title(sprintf('%s - Behavior Ethogram (Top: task, Bottom: spontaneous)', areas{areaIdx}), 'FontSize', 12);
% set(gca, 'YTick', [0.5, 1.5], 'YTickLabel', {'task', 'spontaneous'});
% xlabel('Time (s)', 'FontSize', 12);
% grid on;
% 
% % Save ethogram figure
% saveFileEthogram = fullfile(saveDir, sprintf('d2_task_vs_naturalistic_%s%s_win%d_ethogram.eps', areas{areaIdx}, filenameSuffix, slidingWindowSize));
% exportgraphics(gcf, saveFileEthogram, 'ContentType', 'vector');
% fprintf('Saved ethogram figure to: %s\n', saveFileEthogram);

%% ================================    Statistical Comparison Figure
fprintf('\n=== Creating Statistical Comparison Figure ===\n');

% Extract all valid d2 values (remove NaNs)
d2ReachValid = d2Reach(~isnan(d2Reach));
d2NatValid = d2Nat(~isnan(d2Nat));

% Calculate mean and std
meanReach = mean(d2ReachValid);
stdReach = std(d2ReachValid);
meanNat = mean(d2NatValid);
stdNat = std(d2NatValid);

% Perform statistical test (Wilcoxon rank-sum test for non-parametric comparison)
[pVal, ~, stats] = ranksum(d2ReachValid, d2NatValid);
fprintf('Statistical test: Wilcoxon rank-sum test\n');
fprintf('  Task (reach): mean=%.4f, std=%.4f, n=%d\n', meanReach, stdReach, length(d2ReachValid));
fprintf('  Spontaneous (naturalistic): mean=%.4f, std=%.4f, n=%d\n', meanNat, stdNat, length(d2NatValid));
fprintf('  p-value: %.4e\n', pVal);

% Determine significance marker
if pVal < 0.001
    sigMarker = '***';
elseif pVal < 0.01
    sigMarker = '**';
elseif pVal < 0.05
    sigMarker = '*';
else
    sigMarker = 'ns';
end

% Create bar plot figure
figure(712); clf;
set(gcf, 'Units', 'pixels');
monitorPositions = get(0, 'MonitorPositions');
if size(monitorPositions, 1) >= 2
    targetMonitor = monitorPositions(size(monitorPositions, 1), :);
else
    targetMonitor = monitorPositions(1, :);
end
set(gcf, 'Position', [targetMonitor(1), targetMonitor(2), targetMonitor(3)/3, targetMonitor(4)/2]);

hold on;

% Prepare data for scatter plot
means = [meanReach, meanNat];
stds = [stdReach, stdNat];
groupNames = {'task', 'spontaneous'};

% Create scatter plot with large markers
xPos = 1:length(means);
scatter(xPos(1), means(1), 150, [0 0 1], 'filled', 'o', 'DisplayName', 'Task', ...
    'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
scatter(xPos(2), means(2), 150, [1 0 0], 'filled', 'o', 'DisplayName', 'Spontaneous', ...
    'MarkerEdgeColor', 'k', 'LineWidth', 1.5);

% Add error bars
errorbar(xPos, means, stds, 'k', 'LineWidth', 2, 'LineStyle', 'none', 'CapSize', 10, ...
    'HandleVisibility', 'off');

% Add significance bracket and marker
yMax = max(means + stds);
yMin = min(means - stds);
yRange = yMax - yMin;
% Use maxD2 as upper limit for consistency across areas
yTop = maxD2;
yBracket = yMax + 0.05 * (maxD2 - yMin);

% Draw bracket
plot([1, 1, 2, 2], [yBracket, yTop, yTop, yBracket], 'k-', 'LineWidth', 1.5);

% Add significance marker text
text(1.5, yTop + 0.02 * (maxD2 - yMin), sigMarker, 'HorizontalAlignment', 'center', ...
    'FontSize', 16, 'FontWeight', 'bold');

% Add p-value text
pValStr = sprintf('p = %.3e', pVal);
if pVal >= 0.001
    pValStr = sprintf('p = %.3f', pVal);
end
text(1.5, yBracket - 0.02 * (maxD2 - yMin), pValStr, 'HorizontalAlignment', 'center', ...
    'FontSize', 10);

% Formatting
set(gca, 'XTick', [1, 2], 'XTickLabel', groupNames);
xlim([0.5, 2.5]);
ylabel('d2', 'FontSize', 12);
title(sprintf('%s - Mean d2 Comparison', areas{areaIdx}), 'FontSize', 12);
legend('Location', 'best');
grid on;
ylim([0, maxD2]);

% Save comparison figure
saveFileBar = fullfile(saveDir, sprintf('d2_task_vs_naturalistic_%s%s_win%d_bar_%s.eps', areas{areaIdx}, filenameSuffix, slidingWindowSize, sessionName));
% exportgraphics(gcf, saveFileBar, 'ContentType', 'vector');
saveFileBar = fullfile(saveDir, sprintf('d2_task_vs_naturalistic_%s%s_win%d_bar_%s.png', areas{areaIdx}, filenameSuffix, slidingWindowSize, sessionName));
    exportgraphics(gcf, saveFileBar, 'Resolution', 300);
fprintf('Saved comparison figure to: %s\n', saveFileBar);

%% ================================    Summary Across All Areas
fprintf('\n=== Summary: Mean/Std d2 Across All Areas and Contexts ===\n');
fprintf('%-8s %-15s %-15s %-15s %-15s\n', 'Area', 'task mean', 'task std', 'spontaneous mean', 'spontaneous std');
fprintf('%s\n', repmat('-', 1, 75));

% Initialize summary arrays
summaryMeansTask = zeros(length(areas), 1);
summaryStdsTask = zeros(length(areas), 1);
summaryMeansSpont = zeros(length(areas), 1);
summaryStdsSpont = zeros(length(areas), 1);

% Loop through all brain areas
for iArea = 1:length(areas)
    % Extract d2 for task condition
    d2TaskAll = resultsReach.d2{iArea};
    d2TaskValid = d2TaskAll(~isnan(d2TaskAll));
    
    % Extract d2 for spontaneous condition
    d2SpontAll = resultsNat.d2{iArea};
    d2SpontValid = d2SpontAll(~isnan(d2SpontAll));
    
    % Calculate mean and std
    if ~isempty(d2TaskValid)
        meanTask = mean(d2TaskValid);
        stdTask = std(d2TaskValid);
    else
        meanTask = NaN;
        stdTask = NaN;
    end
    
    if ~isempty(d2SpontValid)
        meanSpont = mean(d2SpontValid);
        stdSpont = std(d2SpontValid);
    else
        meanSpont = NaN;
        stdSpont = NaN;
    end
    
    % Store in summary arrays
    summaryMeansTask(iArea) = meanTask;
    summaryStdsTask(iArea) = stdTask;
    summaryMeansSpont(iArea) = meanSpont;
    summaryStdsSpont(iArea) = stdSpont;
    
    % Print summary for this area
    if ~isnan(meanTask) && ~isnan(meanSpont)
        fprintf('%-8s %-15.4f %-15.4f %-15.4f %-15.4f\n', areas{iArea}, ...
            meanTask, stdTask, meanSpont, stdSpont);
    elseif ~isnan(meanTask)
        fprintf('%-8s %-15.4f %-15.4f %-15s %-15s\n', areas{iArea}, ...
            meanTask, stdTask, 'NaN', 'NaN');
    elseif ~isnan(meanSpont)
        fprintf('%-8s %-15s %-15s %-15.4f %-15.4f\n', areas{iArea}, ...
            'NaN', 'NaN', meanSpont, stdSpont);
    else
        fprintf('%-8s %-15s %-15s %-15s %-15s\n', areas{iArea}, ...
            'NaN', 'NaN', 'NaN', 'NaN');
    end
end

% Print overall summary
fprintf('%s\n', repmat('-', 1, 75));
validTaskMask = ~isnan(summaryMeansTask);
validSpontMask = ~isnan(summaryMeansSpont);

if any(validTaskMask)
    overallMeanTask = mean(summaryMeansTask(validTaskMask));
    overallStdTask = std(summaryMeansTask(validTaskMask));
    fprintf('%-8s %-15.4f %-15.4f %-15s %-15s\n', 'Overall', ...
        overallMeanTask, overallStdTask, '-', '-');
end

if any(validSpontMask)
    overallMeanSpont = mean(summaryMeansSpont(validSpontMask));
    overallStdSpont = std(summaryMeansSpont(validSpontMask));
    if any(validTaskMask)
        fprintf('%-8s %-15s %-15s %-15.4f %-15.4f\n', '', ...
            '-', '-', overallMeanSpont, overallStdSpont);
    else
        fprintf('%-8s %-15s %-15s %-15.4f %-15.4f\n', 'Overall', ...
            '-', '-', overallMeanSpont, overallStdSpont);
    end
end

% ================================    Summary Figure
fprintf('\n=== Creating Summary Figure ===\n');

% Create figure
figure(714); clf;
set(gcf, 'Units', 'pixels');
monitorPositions = get(0, 'MonitorPositions');
if size(monitorPositions, 1) >= 2
    targetMonitor = monitorPositions(size(monitorPositions, 1), :);
else
    targetMonitor = monitorPositions(1, :);
end
set(gcf, 'Position', [targetMonitor(1), targetMonitor(2), targetMonitor(3)/2, targetMonitor(4)/2]);

hold on;

% Prepare data for bar plot
nAreas = length(areas);
xPos = 1:nAreas;

% Prepare data matrix for grouped bar chart
% Keep NaN values - MATLAB bar will skip drawing those bars
barData = [summaryMeansTask, summaryMeansSpont];

% Create grouped bar chart
barColors = [0 0 1; 1 0 0]; % Blue for task, red for spontaneous
b = bar(xPos, barData, 'grouped');

% Set colors using CData property (proper way for 'FaceColor', 'flat')
% CData should be nAreas x 3 (RGB) for each bar series
b(1).CData = repmat(barColors(1, :), nAreas, 1);
b(2).CData = repmat(barColors(2, :), nAreas, 1);

% Add error bars
% Calculate x positions for error bars (centered on each bar group)
xTask = xPos - 0.14;
xSpont = xPos + 0.14;

% Plot error bars for task condition (only for valid data)
validTaskIdx = ~isnan(summaryMeansTask);
if any(validTaskIdx)
    errorbar(xTask(validTaskIdx), summaryMeansTask(validTaskIdx), ...
        summaryStdsTask(validTaskIdx), 'k', 'LineWidth', 1.5, 'LineStyle', 'none', ...
        'CapSize', 5, 'HandleVisibility', 'off');
end

% Plot error bars for spontaneous condition (only for valid data)
validSpontIdx = ~isnan(summaryMeansSpont);
if any(validSpontIdx)
    errorbar(xSpont(validSpontIdx), summaryMeansSpont(validSpontIdx), ...
        summaryStdsSpont(validSpontIdx), 'k', 'LineWidth', 1.5, 'LineStyle', 'none', ...
        'CapSize', 5, 'HandleVisibility', 'off');
end

% Set display names for legend
b(1).DisplayName = 'Task';
b(2).DisplayName = 'Spontaneous';

% Formatting
set(gca, 'XTick', xPos, 'XTickLabel', areas);
xlim([0.5, nAreas + 0.5]);
ylim([0, max(barData(:)*1.3)]);
ylabel('d2', 'FontSize', 12);
xlabel('Brain Area', 'FontSize', 12);
title('Mean d2 Across All Areas and Contexts', 'FontSize', 12);
legend('Location', 'best');
grid on;

% Save summary figure
saveFileSummary = fullfile(saveDir, sprintf('d2_task_vs_naturalistic_summary%s_win%d_%s.eps', filenameSuffix, slidingWindowSize, sessionName));
% exportgraphics(gcf, saveFileSummary, 'ContentType', 'vector');
saveFileSummary = fullfile(saveDir, sprintf('d2_task_vs_naturalistic_summary%s_win%d_%s.png', filenameSuffix, slidingWindowSize, sessionName));
    exportgraphics(gcf, saveFileSummary, 'Resolution', 300);
fprintf('Saved summary figure to: %s\n', saveFileSummary);

fprintf('\n=== Complete ===\n');