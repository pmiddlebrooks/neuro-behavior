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
reachDataFile = fullfile(paths.dropPath, 'reach_data/Copy_of_Y4_100623_Spiketimes_idchan_BEH.mat');
% reachDataFile = fullfile(paths.dropPath, 'reach_data/Y4_06-Oct-2023 14_14_53_NeuroBeh.mat');
reachDataFile = fullfile(paths.dropPath, 'reach_data/AB6_27-Mar-2025 14_04_12_NeuroBeh.mat');

%%
paths = get_paths;

% Determine save directory based on loaded data file name (same as criticality_reach_ar.m)
[~, dataBaseName, ~] = fileparts(reachDataFile);
saveDir = fullfile(paths.dropPath, 'reach_data', dataBaseName);
resultsPathWin = fullfile(saveDir, sprintf('criticality_reach_ar_win%d.mat', slidingWindowSize));

% Load criticality analysis results
fprintf('Loading criticality analysis results from: %s\n', resultsPathWin);
if ~exist(resultsPathWin, 'file')
    error('Results file not found: %s\nMake sure criticality_reach_ar.m has been run for this dataset and window size.', resultsPathWin);
end
results = load(resultsPathWin);
results = results.results;

% Extract areas and parameters
areas = results.areas;
areasToTest = 2:4;
optimalBinSize = results.reach.optimalBinSize;
d2Rea = results.reach.d2;
mrBrRea = results.reach.mrBr;
startS = results.reach.startS;

%% Load reach behavioral data
fprintf('Loading reach behavioral data from: %s\n', reachDataFile);
dataR = load(reachDataFile);

reachStart = dataR.R(:,1); % In seconds
reachStop = dataR.R(:,2);
reachAmp = dataR.R(:,3); % Amplitude of each reach (distance from 0)

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

% Use BlockWithErrors(:,4) if it exists, otherwise use Block(:,5)
if isfield(dataR, 'BlockWithErrors') && size(dataR.BlockWithErrors,2) >= 4
    reachClass = dataR.BlockWithErrors(:,4);
    reachClass = dataR.BlockWithErrors(:,3);
elseif isfield(dataR, 'Block') && size(dataR.Block,2) >= 5
    reachClass = dataR.Block(:,5);
    reachClass = dataR.Block(:,3);
else
    error('Neither BlockWithErrors(:,4) nor Block(:,5) found in dataR.');
end
% reachStartCorr1 = reachStart(ismember(reachClass, 1));
% reachStartCorr2 = reachStart(ismember(reachClass, 2));
% reachStartErr1 = reachStart(ismember(reachClass, [-10 10]));
% reachStartErr2 = reachStart(ismember(reachClass, [-20 20]));
reachStartCorr1 = reachStart(ismember(reachClass, 2));
reachStartCorr2 = reachStart(ismember(reachClass, 4));
reachStartErr1 = reachStart(ismember(reachClass, [1]));
reachStartErr2 = reachStart(ismember(reachClass, [3]));
% % Convert reach start times to frame units for each area
% reachStartFrame = cell(1, length(areas));
% for a = areasToTest
%     % Round reach to nearest frame in criticality analysis bin sizes
%     reachStartFrame{a} = round(reachStart ./ optimalBinSize(a));
% end

%% ==============================================     Peri-Reach Analysis     ==============================================

% Define window parameters
windowDurationSec = 40; % 3-second window around each reach
windowDurationFrames = cell(1, length(areas));
for a = areasToTest
    windowDurationFrames{a} = ceil(windowDurationSec / optimalBinSize(a));
end

% Initialize storage for peri-reach d2/mrBr values (separate correct/error by block)
d2WindowsCorr1 = cell(1, length(areas));
d2WindowsCorr2 = cell(1, length(areas));
d2WindowsErr1 = cell(1, length(areas));
d2WindowsErr2 = cell(1, length(areas));
meanD2PeriReachCorr1 = cell(1, length(areas));
meanD2PeriReachCorr2 = cell(1, length(areas));
meanD2PeriReachErr1 = cell(1, length(areas));
meanD2PeriReachErr2 = cell(1, length(areas));
mrBrWindowsCorr1 = cell(1, length(areas));
mrBrWindowsCorr2 = cell(1, length(areas));
mrBrWindowsErr1 = cell(1, length(areas));
mrBrWindowsErr2 = cell(1, length(areas));
meanMrBrPeriReachCorr1 = cell(1, length(areas));
meanMrBrPeriReachCorr2 = cell(1, length(areas));
meanMrBrPeriReachErr1 = cell(1, length(areas));
meanMrBrPeriReachErr2 = cell(1, length(areas));
timeAxisPeriReach = cell(1, length(areas));

fprintf('\n=== Peri-Reach d2 Criticality Analysis ===\n');

for a = areasToTest
    fprintf('\nProcessing area %s...\n', areas{a});

    % Get d2 values and time points for this area
    d2Values = d2Rea{a};
    timePoints = startS{a};
    % timePoints = startS{a} - results.reach.d2WindowSize(a)/2; % Adjust the times so the leading edge of the analyzed window aligns with reach onset

    % Initialize arrays for this area
    numReachesCorr1 = length(reachStartCorr1);
    numReachesCorr2 = length(reachStartCorr2);
    numReachesErr1 = length(reachStartErr1);
    numReachesErr2 = length(reachStartErr2);
    halfWindow = floor(windowDurationFrames{a} / 2);

    % Create time axis for peri-reach window (centered on reach onset)
    timeAxisPeriReach{a} = (-halfWindow:halfWindow) * optimalBinSize(a);

    % Initialize storage for all reach windows (by block)
    d2WindowsCorr1{a} = nan(numReachesCorr1, windowDurationFrames{a} + 1);
    d2WindowsCorr2{a} = nan(numReachesCorr2, windowDurationFrames{a} + 1);
    d2WindowsErr1{a} = nan(numReachesErr1, windowDurationFrames{a} + 1);
    d2WindowsErr2{a} = nan(numReachesErr2, windowDurationFrames{a} + 1);
    mrBrWindowsCorr1{a} = nan(numReachesCorr1, windowDurationFrames{a} + 1);
    mrBrWindowsCorr2{a} = nan(numReachesCorr2, windowDurationFrames{a} + 1);
    mrBrWindowsErr1{a} = nan(numReachesErr1, windowDurationFrames{a} + 1);
    mrBrWindowsErr2{a} = nan(numReachesErr2, windowDurationFrames{a} + 1);

    % Extract around each correct reach (Block 1)
    validCorr1 = 0;
    for r = 1:numReachesCorr1
        reachTime = reachStartCorr1(r)/1000; % adjust if needed based on units
        [~, closestIdx] = min(abs(timePoints - reachTime));
        winStart = closestIdx - halfWindow;
        winEnd = closestIdx + halfWindow;
        if winStart >= 1 && winEnd <= length(d2Values)
            d2WindowsCorr1{a}(r, :) = d2Values(winStart:winEnd);
            if ~isempty(mrBrRea{a})
                mrBrWindowsCorr1{a}(r, :) = mrBrRea{a}(winStart:winEnd);
            end
            validCorr1 = validCorr1 + 1;
        else
            d2WindowsCorr1{a}(r, :) = nan(1, windowDurationFrames{a} + 1);
            mrBrWindowsCorr1{a}(r, :) = nan(1, windowDurationFrames{a} + 1);
        end
    end

    % Extract around each correct reach (Block 2)
    validCorr2 = 0;
    for r = 1:numReachesCorr2
        reachTime = reachStartCorr2(r)/1000; % adjust if needed based on units
        [~, closestIdx] = min(abs(timePoints - reachTime));
        winStart = closestIdx - halfWindow;
        winEnd = closestIdx + halfWindow;
        if winStart >= 1 && winEnd <= length(d2Values)
            d2WindowsCorr2{a}(r, :) = d2Values(winStart:winEnd);
            if ~isempty(mrBrRea{a})
                mrBrWindowsCorr2{a}(r, :) = mrBrRea{a}(winStart:winEnd);
            end
            validCorr2 = validCorr2 + 1;
        else
            d2WindowsCorr2{a}(r, :) = nan(1, windowDurationFrames{a} + 1);
            mrBrWindowsCorr2{a}(r, :) = nan(1, windowDurationFrames{a} + 1);
        end
    end

    % Extract around each error reach (Block 1)
    validErr1 = 0;
    for r = 1:numReachesErr1
        reachTime = reachStartErr1(r)/1000; % adjust if needed based on units
        [~, closestIdx] = min(abs(timePoints - reachTime));
        winStart = closestIdx - halfWindow;
        winEnd = closestIdx + halfWindow;
        if winStart >= 1 && winEnd <= length(d2Values)
            d2WindowsErr1{a}(r, :) = d2Values(winStart:winEnd);
            if ~isempty(mrBrRea{a})
                mrBrWindowsErr1{a}(r, :) = mrBrRea{a}(winStart:winEnd);
            end
            validErr1 = validErr1 + 1;
        else
            d2WindowsErr1{a}(r, :) = nan(1, windowDurationFrames{a} + 1);
            mrBrWindowsErr1{a}(r, :) = nan(1, windowDurationFrames{a} + 1);
        end
    end

    % Extract around each error reach (Block 2)
    validErr2 = 0;
    for r = 1:numReachesErr2
        reachTime = reachStartErr2(r)/1000; % adjust if needed based on units
        [~, closestIdx] = min(abs(timePoints - reachTime));
        winStart = closestIdx - halfWindow;
        winEnd = closestIdx + halfWindow;
        if winStart >= 1 && winEnd <= length(d2Values)
            d2WindowsErr2{a}(r, :) = d2Values(winStart:winEnd);
            if ~isempty(mrBrRea{a})
                mrBrWindowsErr2{a}(r, :) = mrBrRea{a}(winStart:winEnd);
            end
            validErr2 = validErr2 + 1;
        else
            d2WindowsErr2{a}(r, :) = nan(1, windowDurationFrames{a} + 1);
            mrBrWindowsErr2{a}(r, :) = nan(1, windowDurationFrames{a} + 1);
        end
    end

    % Calculate mean d2/mrBr values across all valid reaches
    meanD2PeriReachCorr1{a} = nanmean(d2WindowsCorr1{a}, 1);
    meanD2PeriReachCorr2{a} = nanmean(d2WindowsCorr2{a}, 1);
    meanD2PeriReachErr1{a} = nanmean(d2WindowsErr1{a}, 1);
    meanD2PeriReachErr2{a} = nanmean(d2WindowsErr2{a}, 1);
    meanMrBrPeriReachCorr1{a} = nanmean(mrBrWindowsCorr1{a}, 1);
    meanMrBrPeriReachCorr2{a} = nanmean(mrBrWindowsCorr2{a}, 1);
    meanMrBrPeriReachErr1{a} = nanmean(mrBrWindowsErr1{a}, 1);
    meanMrBrPeriReachErr2{a} = nanmean(mrBrWindowsErr2{a}, 1);

    fprintf('Area %s: Corr B1=%d, Corr B2=%d, Err B1=%d, Err B2=%d valid reaches\n', areas{a}, validCorr1, validCorr2, validErr1, validErr2);
end

% ==============================================     Plotting Results     ==============================================

% Toggle plotting error reaches (single traces and mean)
plotErrors = true;

% Create peri-reach plots for each area: 2 (metrics) x numAreas
figure(300 + slidingWindowSize); clf;
set(gcf, 'Position', [100, 100, 1400, 700]);

% Use tight_subplot for layout with dynamic columns
numCols = length(areasToTest);
ha = tight_subplot(2, numCols, [0.08 0.04], [0.08 0.1], [0.06 0.04]);

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

for idx = 1:length(areasToTest)
    a = areasToTest(idx);
    % Top row: d2
    axes(ha(idx));
    hold on;

    % % Plot individual reach windows (all single traces in light gray)
    % for r = 1:size(d2WindowsCorr{a}, 1)
    %     if ~all(isnan(d2WindowsCorr{a}(r, :)))
    %         plot(timeAxisPeriReach{a}, d2WindowsCorr{a}(r, :), 'Color', [.7 .7 .7], 'LineWidth', 0.5);
    %     end
    % end
    % if plotErrors
    %     for r = 1:size(d2WindowsErr{a}, 1)
    %         if ~all(isnan(d2WindowsErr{a}(r, :)))
    %             plot(timeAxisPeriReach{a}, d2WindowsErr{a}(r, :), 'Color', [.7 .7 .7], 'LineWidth', 0.5);
    %         end
    %     end
    % end

    % Calculate SEM for mean traces (by block)
    semCorr1 = nanstd(d2WindowsCorr1{a}, 0, 1) / max(1, sqrt(sum(~all(isnan(d2WindowsCorr1{a}), 2))));
    semCorr2 = nanstd(d2WindowsCorr2{a}, 0, 1) / max(1, sqrt(sum(~all(isnan(d2WindowsCorr2{a}), 2))));
    if plotErrors
        semErr1 = nanstd(d2WindowsErr1{a}, 0, 1) / max(1, sqrt(sum(~all(isnan(d2WindowsErr1{a}), 2))));
        semErr2 = nanstd(d2WindowsErr2{a}, 0, 1) / max(1, sqrt(sum(~all(isnan(d2WindowsErr2{a}), 2))));
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
    
    % Plot mean d2 values per block: B1 solid, B2 dash-dot; errors dashed variants
    hCorr1 = plot(timeAxisPeriReach{a}, meanD2PeriReachCorr1{a}, 'Color', colCorr1, 'LineWidth', 3, 'LineStyle', '-');
    hCorr2 = plot(timeAxisPeriReach{a}, meanD2PeriReachCorr2{a}, 'Color', colCorr2, 'LineWidth', 3, 'LineStyle', '-');
    if plotErrors
        hErr1 = plot(timeAxisPeriReach{a}, meanD2PeriReachErr1{a}, 'Color', colCorr1, 'LineWidth', 2, 'LineStyle', '--');
        hErr2 = plot(timeAxisPeriReach{a}, meanD2PeriReachErr2{a}, 'Color', colCorr2, 'LineWidth', 2, 'LineStyle', '--');
    end

    % Add vertical line at reach onset
    plot([0 0], ylim, 'k--', 'LineWidth', 2);

    % Formatting
    xlabel('Time relative to reach onset (s)', 'FontSize', 12);
    ylabel('d2 Criticality', 'FontSize', 12);
    title(sprintf('%s - Peri-Reach d2 Criticality (Window: %gs)', areas{a}, slidingWindowSize), 'FontSize', 14);
    grid on;

    % Set x-axis ticks and limits in seconds
    xMin = min(timeAxisPeriReach{a});
    xMax = max(timeAxisPeriReach{a});
    xlim([xMin xMax]);
    xTicks = ceil(xMin):floor(xMax);
    if isempty(xTicks)
        % Fallback to 5 ticks across range
        xTicks = linspace(xMin, xMax, 5);
    end
    xticks(xTicks);
    xticklabels(string(xTicks));

    % Set consistent y-axis limits and ticks across all subplots
    ylim(yLimCommon);
    yTicks = linspace(yLimCommon(1), yLimCommon(2), 5);
    yticks(yTicks);
    yticklabels(string(round(yTicks, 3)));

    % Add legend for mean traces
    % Legend entries per block
    legs = [hCorr1 hCorr2];
    leg = {'d2 Correct B1', 'd2 Correct B2'};
    if plotErrors
        legs = [legs hErr1 hErr2];
        leg = [leg {'d2 Error B1', 'd2 Error B2'}];
    end
    legend(legs, leg, 'Location', 'best', 'FontSize', 9);

    % Bottom row: mrBr
    axes(ha(numCols + idx));
    hold on;

    % Compute SEM for mrBr (by block)
    baseColor = colorSpecToRGB(colors{a});
    colCorr1 = baseColor;
    colCorr2 = min(1, baseColor + 0.3);
    semMrCorr1 = nanstd(mrBrWindowsCorr1{a}, 0, 1) / max(1, sqrt(sum(~all(isnan(mrBrWindowsCorr1{a}), 2))));
    semMrCorr2 = nanstd(mrBrWindowsCorr2{a}, 0, 1) / max(1, sqrt(sum(~all(isnan(mrBrWindowsCorr2{a}), 2))));
    if plotErrors
        semMrErr1 = nanstd(mrBrWindowsErr1{a}, 0, 1) / max(1, sqrt(sum(~all(isnan(mrBrWindowsErr1{a}), 2))));
        semMrErr2 = nanstd(mrBrWindowsErr2{a}, 0, 1) / max(1, sqrt(sum(~all(isnan(mrBrWindowsErr2{a}), 2))));
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

    % Formatting
    plot([0 0], ylim, 'k--', 'LineWidth', 2);
    xlabel('Time relative to reach onset (s)', 'FontSize', 12);
    ylabel('MR Branching Ratio', 'FontSize', 12);
    title(sprintf('%s - Peri-Reach mrBr (Window: %gs)', areas{a}, slidingWindowSize), 'FontSize', 14);
    grid on;
    xMin = min(timeAxisPeriReach{a}); xMax = max(timeAxisPeriReach{a});
    xlim([xMin xMax]);
    xTicks = ceil(xMin):floor(xMax);
    if isempty(xTicks), xTicks = linspace(xMin, xMax, 5); end
    xticks(xTicks); xticklabels(string(xTicks));
    ylim([0 1.05]);
    % Reference line at mrBr = 1
    yline(1, 'k:', 'LineWidth', 1.5);
    
    % Ensure y-tick labels are visible
    set(gca, 'YTickLabelMode', 'auto');

    % Legend for mrBr
    if plotErrors
        legend([hMrCorr1 hMrCorr2 hMrErr1 hMrErr2], {'mrBr Correct B1','mrBr Correct B2','mrBr Error B1','mrBr Error B2'}, 'Location', 'best', 'FontSize', 9);
    else
        legend([hMrCorr1 hMrCorr2], {'mrBr Correct B1','mrBr Correct B2'}, 'Location', 'best', 'FontSize', 9);
    end
end

sgtitle(sprintf('Peri-Reach d2 (top) and mrBr (bottom) (Sliding Window: %gs)', slidingWindowSize), 'FontSize', 16);

% Save combined figure (in same data-specific folder)
filename = fullfile(saveDir, sprintf('peri_reach_d2_mrbr_win%gs.png', slidingWindowSize));
exportgraphics(gcf, filename, 'Resolution', 300);
fprintf('Saved peri-reach d2+mrBr plot to: %s\n', filename);

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
