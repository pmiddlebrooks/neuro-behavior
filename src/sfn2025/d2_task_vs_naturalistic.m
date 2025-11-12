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
areaIdx = 1; % Brain area: 1=M23, 2=M56, 3=DS, 4=VS

% Sliding window size (should match the one used in criticality_sliding_window_ar.m)
slidingWindowSize = 20; % seconds

% Setup paths
paths = get_paths;
areas = {'M23', 'M56', 'DS', 'VS'};

% Reach data file (should match the one used in criticality_sliding_window_ar.m)
reachDataFile = fullfile(paths.reachDataPath, 'Y4_06-Oct-2023 14_14_53_NeuroBeh.mat');
[~, dataBaseName, ~] = fileparts(reachDataFile);

% Naturalistic data parameters (should match criticality_sliding_window_ar.m)
animal = 'ag25290';
sessionBhv = '112321_1';


% ================================    Load criticality results
% Load reach criticality results
fprintf('\n=== Loading Reach Criticality Results ===\n');
reachSaveDir = fullfile(paths.dropPath, 'reach_task/results', dataBaseName);
reachResultsPath = fullfile(reachSaveDir, sprintf('criticality_sliding_window_ar_win%d.mat', slidingWindowSize));

if ~exist(reachResultsPath, 'file')
    error('Reach results file not found: %s\nRun criticality_sliding_window_ar.m for reach data first.', reachResultsPath);
end

resultsReach = load(reachResultsPath);
resultsReach = resultsReach.results;
fprintf('Loaded reach results for area %s\n', areas{areaIdx});

% Load naturalistic criticality results
fprintf('\n=== Loading Naturalistic Criticality Results ===\n');
natSaveDir = fullfile(paths.dropPath, 'criticality/results');
natResultsPath = fullfile(natSaveDir, sprintf('criticality_sliding_window_ar_win%d.mat', slidingWindowSize));

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

%% ================================    Analysis   ================================
% Extract d2 data for specified area
d2Reach = resultsReach.d2{areaIdx};
startSReach = resultsReach.startS{areaIdx};
d2Nat = resultsNat.d2{areaIdx};
startSNat = resultsNat.startS{areaIdx};

% Find common time range
timeStart = min([min(startSReach), min(startSNat), 0]);
timeEnd = max([max(startSReach), max(startSNat), max(timeAxisReachBhv), max(timeAxisNatBhv)]);

% Create figure
fprintf('\n=== Creating Figure ===\n');
figure(711); clf;
set(gcf, 'Units', 'pixels');
monitorPositions = get(0, 'MonitorPositions');
if size(monitorPositions, 1) >= 2
    targetMonitor = monitorPositions(size(monitorPositions, 1), :);
else
    targetMonitor = monitorPositions(1, :);
end
set(gcf, 'Position', [targetMonitor(1), targetMonitor(2), targetMonitor(3), targetMonitor(4)/2]);

% Use tight_subplot for layout
ha = tight_subplot(2, 1, [0.08 0.04], [0.08 0.1], [0.06 0.04]);

% Top subplot: Ethogram
axes(ha(1));
hold on;

% Plot reach behavior ethogram
for bhvCode = 1:6
    bhvMask = (bhvIDReach == bhvCode);
    if any(bhvMask)
        % Create color blocks by finding contiguous regions
        diffMask = diff([0; bhvMask; 0]);
        starts = find(diffMask == 1);
        ends = find(diffMask == -1) - 1;
        
        for b = 1:length(starts)
            xStart = timeAxisReachBhv(starts(b));
            xEnd = timeAxisReachBhv(ends(b));
            fill([xStart, xEnd, xEnd, xStart], [0, 0, 1, 1], colorsReach(bhvCode, :), ...
                'EdgeColor', 'none', 'DisplayName', behaviorsReach{bhvCode});
        end
    end
end

% Plot naturalistic behavior ethogram (offset vertically)
for bhvCode = unique(bhvIDNat)'
    if bhvCode >= 0 % Skip invalid labels
        bhvMask = (bhvIDNat == bhvCode);
        if any(bhvMask)
            % Create color blocks by finding contiguous regions
            diffMask = diff([0; bhvMask; 0]);
            starts = find(diffMask == 1);
            ends = find(diffMask == -1) - 1;
            
            % Find color index
            colorIdx = find(codesNat == bhvCode, 1);
            if ~isempty(colorIdx) && colorIdx <= size(colorsNat, 1)
                plotColor = colorsNat(colorIdx, :);
            else
                plotColor = [0.5, 0.5, 0.5]; % Gray for unmapped
            end
            
            for b = 1:length(starts)
                xStart = timeAxisNatBhv(starts(b));
                xEnd = timeAxisNatBhv(ends(b));
                fill([xStart, xEnd, xEnd, xStart], [1, 1, 2, 2], plotColor, ...
                    'EdgeColor', 'none');
            end
        end
    end
end

xlim([timeStart, timeEnd]);
ylim([0, 2]);
ylabel('Behavior', 'FontSize', 12);
title(sprintf('%s - Behavior Ethogram (Top: Reach, Bottom: Naturalistic)', areas{areaIdx}), 'FontSize', 12);
set(gca, 'YTick', [0.5, 1.5], 'YTickLabel', {'Reach', 'Naturalistic'});
grid on;

% Bottom subplot: d2 time series
axes(ha(2));
hold on;

% Plot reach d2
if ~isempty(d2Reach) && ~all(isnan(d2Reach))
    plot(startSReach, d2Reach, 'b-', 'LineWidth', 2, 'DisplayName', 'Reach d2');
end

% Plot naturalistic d2
if ~isempty(d2Nat) && ~all(isnan(d2Nat))
    plot(startSNat, d2Nat, 'r-', 'LineWidth', 2, 'DisplayName', 'Naturalistic d2');
end

% Plot reach start markers (filled green circles)
reachStartInRange = reachStart(reachStart >= timeStart & reachStart <= timeEnd);
if ~isempty(reachStartInRange)
    % Get d2 values at reach start times (interpolate if needed)
    d2AtReach = interp1(startSReach, d2Reach, reachStartInRange, 'linear', 'extrap');
    scatter(reachStartInRange, d2AtReach, 50, 'g', 'filled', 'o', ...
        'DisplayName', 'Reach Start', 'MarkerEdgeColor', 'k', 'LineWidth', 1);
end

xlim([timeStart, timeEnd]);
xlabel('Time (s)', 'FontSize', 12);
ylabel('d2', 'FontSize', 12);
title(sprintf('%s - d2 Criticality Comparison', areas{areaIdx}), 'FontSize', 12);
legend('Location', 'best');
grid on;

% Save figure
saveDir = fullfile(paths.dropPath, 'sfn2025', 'd2_comparison');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end
saveFile = fullfile(saveDir, sprintf('d2_task_vs_naturalistic_%s_win%d.eps', areas{areaIdx}, slidingWindowSize));
exportgraphics(gcf, saveFile, 'ContentType', 'vector');
fprintf('Saved figure to: %s\n', saveFile);

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

% Prepare data for bar plot
means = [meanReach, meanNat];
stds = [stdReach, stdNat];
groupNames = {'Task', 'Spontaneous'};

% Create bar plot
b = bar(means, 'FaceColor', 'flat');
b.CData(1,:) = [0 0 1]; % Blue for task
b.CData(2,:) = [1 0 0]; % Red for spontaneous

% Add error bars
xPos = 1:length(means);
errorbar(xPos, means, stds, 'k', 'LineWidth', 2, 'LineStyle', 'none', 'CapSize', 10);

% Add significance bracket and marker
yMax = max(means + stds);
yMin = min(means - stds);
yRange = yMax - yMin;
yTop = yMax + 0.1 * yRange;
yBracket = yMax + 0.05 * yRange;

% Draw bracket
plot([1, 1, 2, 2], [yBracket, yTop, yTop, yBracket], 'k-', 'LineWidth', 1.5);

% Add significance marker text
text(1.5, yTop + 0.02 * yRange, sigMarker, 'HorizontalAlignment', 'center', ...
    'FontSize', 16, 'FontWeight', 'bold');

% Add p-value text
pValStr = sprintf('p = %.3e', pVal);
if pVal >= 0.001
    pValStr = sprintf('p = %.3f', pVal);
end
text(1.5, yBracket - 0.02 * yRange, pValStr, 'HorizontalAlignment', 'center', ...
    'FontSize', 10);

% Formatting
set(gca, 'XTick', [1, 2], 'XTickLabel', groupNames);
xlim([0.5, 2.5]);
ylabel('d2', 'FontSize', 12);
title(sprintf('%s - Mean d2 Comparison', areas{areaIdx}), 'FontSize', 12);
grid on;
ylim([yMin - 0.15 * yRange, yTop + 0.1 * yRange]);

% Save bar plot figure
saveFileBar = fullfile(saveDir, sprintf('d2_task_vs_naturalistic_%s_win%d_bar.eps', areas{areaIdx}, slidingWindowSize));
exportgraphics(gcf, saveFileBar, 'ContentType', 'vector');
fprintf('Saved bar plot figure to: %s\n', saveFileBar);

fprintf('\n=== Complete ===\n');